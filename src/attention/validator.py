import torch


def validate_constructor_args(cls):
    def validate(embed_dim, num_head):
        assert embed_dim % num_head == 0, "embed_dim must be divisible by num_head"
        return cls(embed_dim, num_head)
    return validate

def validate_forward_input_args(fn):
    def validate_args(self, 
                      key: torch.Tensor, 
                      query: torch.Tensor, 
                      value: torch.Tensor,
                      mask=False):
        assert len(key.shape) == 3, "key shape is inappropriate. expected (N, S, E) but got {}".format(key.shape)
        assert len(query.shape) == 3, "query shape is inappropriate. expected (N, S, E) but got {}".format(query.shape)
        assert len(value.shape) == 3, "value shape is inappropriate. expected (N, S, E) but got {}".format(value.shape)
        assert key.shape[2] == query.shape[2] == value.shape[2] == self.embed_dim, "key, query, value must have the same embedding dimension: {} {} {} {}".format(key.shape, query.shape, value.shape, self.embed_dim)
        assert key.shape[1] == value.shape[1], "key, value must have the same sequence length: {} {}".format(key.shape, value.shape)
        return fn(self, key, query, value)
    return validate_args