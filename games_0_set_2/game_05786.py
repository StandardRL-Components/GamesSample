
# Generated: 2025-08-28T06:06:04.232539
# Source Brief: brief_05786.md
# Brief Index: 5786

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game objects
class Block:
    """Represents a single breakable block."""
    def __init__(self, rect, color, points):
        self.rect = rect
        self.color = color
        self.points = points

class Particle:
    """Represents a particle for visual effects."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-2.5, -0.5)
        self.color = color
        self.life = random.randint(20, 40)
        self.max_life = self.life

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # Gravity
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = max(0, min(255, int(255 * (self.life / self.max_life))))
            radius = int(3 * (self.life / self.max_life))
            if radius > 0:
                # Use a temporary surface for alpha blending
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.color, alpha), (radius, radius), radius)
                surface.blit(s, (int(self.x) - radius, int(self.y) - radius))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "A fast-paced, procedurally generated block breaker where strategic paddle positioning "
        "is key to maximizing score and surviving the blitz."
    )

    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 8
    BALL_RADIUS = 8
    MAX_LIVES = 3
    MAX_LEVELS = 3
    MAX_STEPS = 2500

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_PADDLE = (240, 240, 240)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (255, 255, 150)
    COLOR_TEXT = (255, 255, 255)
    
    BLOCK_COLORS = {
        10: (200, 50, 50),   # Red
        20: (50, 200, 50),   # Green
        30: (50, 100, 200),  # Blue
        50: (220, 120, 0),   # Orange
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        self.render_mode = render_mode
        self.game_state = {} # To hold all mutable game state
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game_state = {
            "steps": 0,
            "score": 0,
            "game_over": False,
            "win": False,
            "lives": self.MAX_LIVES,
            "level": 1,
            "paddle_rect": pygame.Rect(
                (self.WIDTH - self.PADDLE_WIDTH) / 2,
                self.HEIGHT - self.PADDLE_HEIGHT - 10,
                self.PADDLE_WIDTH,
                self.PADDLE_HEIGHT
            ),
            "ball_pos": np.array([0.0, 0.0]),
            "ball_vel": np.array([0.0, 0.0]),
            "ball_base_speed": 4.0,
            "ball_on_paddle": True,
            "blocks": [],
            "particles": [],
            "frames_since_last_hit": 0
        }
        self._generate_blocks()
        self._reset_ball()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # --- Handle Actions ---
        movement = action[0]
        space_held = action[1] == 1

        if movement in [3, 4]: # Left or Right
            move_dir = -1 if movement == 3 else 1
            self.game_state["paddle_rect"].x += move_dir * self.PADDLE_SPEED
            self.game_state["paddle_rect"].clamp_ip(self.screen.get_rect())
            reward -= 0.02 # Small penalty for movement

        if self.game_state["ball_on_paddle"]:
            if space_held:
                self.game_state["ball_on_paddle"] = False
                # Launch ball
                angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4)
                speed = self.game_state["ball_base_speed"] + (self.game_state["level"] - 1) * 0.4
                self.game_state["ball_vel"] = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
                # SFX: Ball launch
            else:
                # Ball follows paddle
                self.game_state["ball_pos"][0] = self.game_state["paddle_rect"].centerx
                self.game_state["ball_pos"][1] = self.game_state["paddle_rect"].top - self.BALL_RADIUS
        
        # --- Update Game Logic ---
        if not self.game_state["ball_on_paddle"]:
            self._update_ball()
            reward += self._handle_collisions()

        self._update_particles()
        
        # --- Check Game State ---
        if not self.game_state["blocks"]:
            reward += 10 # Level complete reward
            self.game_state["level"] += 1
            if self.game_state["level"] > self.MAX_LEVELS:
                self.game_state["game_over"] = True
                self.game_state["win"] = True
                reward += 100 # Win game reward
            else:
                self._generate_blocks()
                self._reset_ball()
                # SFX: Level up

        if self.game_state["lives"] <= 0:
            self.game_state["game_over"] = True
            reward -= 100 # Lose game penalty

        self.game_state["steps"] += 1
        terminated = self.game_state["game_over"] or self.game_state["steps"] >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _update_ball(self):
        gs = self.game_state
        gs["ball_pos"] += gs["ball_vel"]
        gs["frames_since_last_hit"] += 1

        # Anti-softlock
        if gs["frames_since_last_hit"] > 200:
            self._reset_ball()
            gs["lives"] -= 1
            # No reward change for this, it's a bug fix

    def _handle_collisions(self):
        gs = self.game_state
        reward = 0
        ball_rect = pygame.Rect(gs["ball_pos"][0] - self.BALL_RADIUS, gs["ball_pos"][1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            gs["ball_vel"][0] *= -1
            ball_rect.clamp_ip(self.screen.get_rect())
            gs["ball_pos"][0] = ball_rect.centerx
            # SFX: Wall bounce

        if ball_rect.top <= 0:
            gs["ball_vel"][1] *= -1
            ball_rect.clamp_ip(self.screen.get_rect())
            gs["ball_pos"][1] = ball_rect.centery
            # SFX: Wall bounce

        # Bottom wall (miss)
        if ball_rect.top >= self.HEIGHT:
            gs["lives"] -= 1
            reward -= 10
            self._reset_ball()
            # SFX: Life lost
            return reward

        # Paddle collision
        if ball_rect.colliderect(gs["paddle_rect"]) and gs["ball_vel"][1] > 0:
            gs["ball_vel"][1] *= -1
            
            # Change angle based on hit position
            offset = (ball_rect.centerx - gs["paddle_rect"].centerx) / (gs["paddle_rect"].width / 2)
            gs["ball_vel"][0] += offset * 2.0
            
            # Normalize speed
            current_speed = np.linalg.norm(gs["ball_vel"])
            target_speed = gs["ball_base_speed"] + (gs["level"] - 1) * 0.4
            gs["ball_vel"] = (gs["ball_vel"] / current_speed) * target_speed
            
            gs["frames_since_last_hit"] = 0
            ball_rect.bottom = gs["paddle_rect"].top
            gs["ball_pos"][1] = ball_rect.centery
            # SFX: Paddle hit
        
        # Block collisions
        for block in gs["blocks"][:]:
            if ball_rect.colliderect(block.rect):
                reward += 0.1 # Hit block reward
                
                # Determine bounce direction
                if ball_rect.centery < block.rect.top or ball_rect.centery > block.rect.bottom:
                    gs["ball_vel"][1] *= -1
                else:
                    gs["ball_vel"][0] *= -1

                reward += 1.0 # Destroy block reward
                gs["score"] += block.points
                self._create_particles(block.rect.center, block.color)
                gs["blocks"].remove(block)
                gs["frames_since_last_hit"] = 0
                # SFX: Block break
                break # Only break one block per frame
        
        return reward

    def _reset_ball(self):
        self.game_state["ball_on_paddle"] = True
        self.game_state["frames_since_last_hit"] = 0
        self.game_state["ball_vel"] = np.array([0.0, 0.0])
        # Position will be set relative to paddle in the main loop

    def _generate_blocks(self):
        self.game_state["blocks"] = []
        rows = 5
        cols = 10
        block_width = self.WIDTH // cols
        block_height = 20
        points_map = [50, 30, 30, 20, 10]

        for r in range(rows):
            for c in range(cols):
                # Simple procedural generation based on level
                if self.np_random.random() < 0.2 + (self.game_state["level"] * 0.1):
                    continue
                
                points = points_map[r]
                color = self.BLOCK_COLORS[points]
                rect = pygame.Rect(c * block_width, r * block_height + 50, block_width-1, block_height-1)
                self.game_state["blocks"].append(Block(rect, color, points))

    def _create_particles(self, pos, color):
        for _ in range(15):
            self.game_state["particles"].append(Particle(pos[0], pos[1], color))

    def _update_particles(self):
        for p in self.game_state["particles"][:]:
            p.update()
            if p.life <= 0:
                self.game_state["particles"].remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG_BOTTOM)
        self._render_background()
        
        self._render_blocks()
        self._render_paddle()
        self._render_ball()
        self._render_particles()
        
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        # Draw a gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_blocks(self):
        for block in self.game_state["blocks"]:
            pygame.draw.rect(self.screen, block.color, block.rect)

    def _render_paddle(self):
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.game_state["paddle_rect"], border_radius=3)

    def _render_ball(self):
        pos = (int(self.game_state["ball_pos"][0]), int(self.game_state["ball_pos"][1]))
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS + 4, (*self.COLOR_BALL_GLOW, 50))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_particles(self):
        for p in self.game_state["particles"]:
            p.draw(self.screen)

    def _render_ui(self):
        # Score
        score_surf = self.font_small.render(f"SCORE: {self.game_state['score']}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Level
        level_surf = self.font_small.render(f"LEVEL: {self.game_state['level']}", True, self.COLOR_TEXT)
        level_rect = level_surf.get_rect(centerx=self.WIDTH/2, y=10)
        self.screen.blit(level_surf, level_rect)

        # Lives
        for i in range(self.game_state["lives"]):
            pos = (self.WIDTH - 20 - i * 20, 20)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 6, self.COLOR_PADDLE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 6, self.COLOR_PADDLE)
        
        # Game Over / Win message
        if self.game_state["game_over"]:
            message = "YOU WIN!" if self.game_state["win"] else "GAME OVER"
            color = (100, 255, 100) if self.game_state["win"] else (255, 100, 100)
            msg_surf = self.font_large.render(message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.game_state.get("score", 0),
            "steps": self.game_state.get("steps", 0),
            "level": self.game_state.get("level", 1),
            "lives": self.game_state.get("lives", 0),
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to 'human' to visualize, or 'rgb_array' for headless
    render_mode = "human" 

    if render_mode == "human":
        import os
        # Ensure the display is available for human rendering
        if os.environ.get("SDL_VIDEODRIVER", "") != "dummy":
            pygame.display.set_caption("Block Breaker")
            screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    else:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode=render_mode)
    obs, info = env.reset()
    
    terminated = False
    total_reward = 0
    
    # --- Manual Control Mapping ---
    # This maps keyboard keys to the MultiDiscrete action space
    key_to_action = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Main game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op action

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Movement action
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Space action
        if keys[pygame.K_SPACE]:
            action[1] = 1

        # Shift action (unused in this game)
        if keys[pygame.K_SHIFT]:
            action[2] = 1
            
        if terminated:
            # If the game is over, wait for a key press to reset
            if any(keys):
                print(f"Resetting game. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                obs, info = env.reset()
                terminated = False
                total_reward = 0
        else:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        if render_mode == "human":
            # Convert observation back to a surface pygame can display
            # Transpose from (H, W, C) to (W, H, C)
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(60) # Limit to 60 FPS for human play

    env.close()