
# Generated: 2025-08-28T02:08:35.909289
# Source Brief: brief_04354.md
# Brief Index: 4354

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# To ensure Pygame runs headlessly if needed
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = "Controls: ←→ to move the paddle. Press space to launch the ball."

    # User-facing game description
    game_description = "A fast-paced, procedurally generated block-breaking game where strategic paddle positioning and risky plays are rewarded."

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_combo = pygame.font.SysFont("Consolas", 24, bold=True)

        # Game constants
        self.MAX_STEPS = 5000
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 50, 20
        self.INITIAL_BALLS = 3
        self.MAX_STAGES = 3

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (30, 40, 80)
        self.COLOR_PADDLE = (240, 240, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = [
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 165, 0),  # Orange
            (0, 255, 0),    # Green
            (255, 69, 0),   # Red-Orange
        ]

        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.ball_base_speed = None
        self.blocks = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.balls_left = None
        self.stage = None
        self.combo = None

        # This will be properly initialized in reset()
        self.np_random = None

        self.reset()
        
        # Run validation check on initialization
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = self.INITIAL_BALLS
        self.stage = 1
        self.combo = 0
        self.particles = []

        self._setup_stage()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for time to encourage efficiency

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            
            # --- UPDATE GAME LOGIC ---
            self._update_paddle(movement)
            reward += self._update_ball(space_held)
            self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info(),
        )

    def _setup_stage(self):
        """Initializes the game state for the current stage."""
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.ball_launched = False
        self.ball_base_speed = 5.0 + (self.stage - 1) * 0.5
        self._reset_ball()
        self._generate_blocks()

    def _reset_ball(self):
        """Resets the ball to the paddle."""
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_launched = False
        self.combo = 0 # Reset combo on ball loss

    def _generate_blocks(self):
        """Procedurally generates blocks for the current stage."""
        self.blocks = []
        rows = 4 + self.stage
        cols = 10
        block_count_target = 30 + (self.stage - 1) * 20
        
        x_spacing = (self.BLOCK_WIDTH + 4)
        y_spacing = (self.BLOCK_HEIGHT + 4)
        x_offset = (self.WIDTH - (cols * x_spacing)) // 2 + 2
        y_offset = 50

        available_pos = []
        for r in range(rows):
            for c in range(cols):
                available_pos.append((c, r))
        
        self.np_random.shuffle(available_pos)

        for i in range(min(block_count_target, len(available_pos))):
            c, r = available_pos[i]
            x = x_offset + c * x_spacing
            y = y_offset + r * y_spacing
            block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
            color = self.np_random.choice(self.BLOCK_COLORS)
            self.blocks.append({"rect": block_rect, "color": color})

    def _update_paddle(self, movement):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        # If ball is not launched, it follows the paddle
        if not self.ball_launched:
            self.ball_pos.x = self.paddle.centerx

    def _update_ball(self, space_held):
        """Updates ball position and handles collisions."""
        reward = 0
        if not self.ball_launched and space_held:
            self.ball_launched = True
            angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4)
            self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.ball_base_speed
            # sfx: launch_ball

        if not self.ball_launched:
            return reward

        self.ball_pos += self.ball_vel

        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball_pos.x))
            # sfx: wall_bounce
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
            reward -= 0.2 # Penalty for hitting top wall
            # sfx: wall_bounce
        
        # Paddle collision
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        if self.paddle.colliderect(ball_rect) and self.ball_vel.y > 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
            
            # Change horizontal velocity based on where it hit the paddle
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x = self.ball_base_speed * offset * 1.5
            self.ball_vel.normalize_ip()
            self.ball_vel *= self.ball_base_speed
            
            reward += 0.1 # Reward for keeping ball in play
            self.combo = 0 # Reset combo on paddle hit
            # sfx: paddle_hit

        # Block collisions
        for block in self.blocks[:]:
            if block["rect"].colliderect(ball_rect):
                self._spawn_particles(block["rect"].center, block["color"])
                self.blocks.remove(block)
                
                # Determine bounce direction
                if abs(self.ball_pos.y - block["rect"].centery) > self.BLOCK_HEIGHT / 2:
                    self.ball_vel.y *= -1
                else:
                    self.ball_vel.x *= -1

                self.combo += 1
                reward += 1.0 + (0.5 * (self.combo - 1))
                self.score += 10 * self.combo
                # sfx: block_break
                break

        # Ball loss
        if self.ball_pos.y > self.HEIGHT:
            self.balls_left -= 1
            reward -= 1.0
            if self.balls_left > 0:
                self._reset_ball()
                # sfx: lose_ball
            else:
                self.game_over = True
                # sfx: game_over
        
        # Stage clear
        if not self.blocks and not self.game_over:
            self.stage += 1
            reward += 10.0
            if self.stage > self.MAX_STAGES:
                self.game_over = True
                reward += 100.0 # Win bonus
            else:
                self._setup_stage()
                # sfx: stage_clear

        return reward

    def _spawn_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Damping
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return self.balls_left <= 0 or self.steps >= self.MAX_STEPS or (self.stage > self.MAX_STAGES)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "balls_left": self.balls_left,
            "combo": self.combo
        }

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        """Draws a vertical gradient for the background."""
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block["color"]), block["rect"], 2)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        
        # Draw particles
        for p in self.particles:
            life_ratio = p["lifespan"] / p["max_lifespan"]
            radius = int(life_ratio * 4)
            if radius > 0:
                alpha = int(life_ratio * 255)
                color = (p["color"][0], p["color"][1], p["color"][2], alpha)
                
                # Use a temporary surface for alpha blending
                particle_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (radius, radius), radius)
                self.screen.blit(particle_surf, (int(p["pos"].x - radius), int(p["pos"].y - radius)))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score:06}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Stage
        stage_text = self.font_ui.render(f"STAGE: {self.stage}", True, self.COLOR_UI_TEXT)
        stage_rect = stage_text.get_rect(center=(self.WIDTH // 2, 10 + score_text.get_height() // 2))
        self.screen.blit(stage_text, stage_rect)

        # Balls left
        balls_text = self.font_ui.render(f"BALLS: {self.balls_left}", True, self.COLOR_UI_TEXT)
        balls_rect = balls_text.get_rect(right=self.WIDTH - 10, top=10)
        self.screen.blit(balls_text, balls_rect)

        # Combo
        if self.combo > 1:
            combo_text = self.font_combo.render(f"x{self.combo}", True, self.COLOR_BALL)
            pos_x = self.paddle.centerx
            pos_y = self.paddle.top - 40
            combo_rect = combo_text.get_rect(center=(pos_x, pos_y))
            self.screen.blit(combo_text, combo_rect)
            
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
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
if __name__ == "__main__":
    env = GameEnv()
    env.reset()
    
    # To display the game, we need to un-dummy the video driver and create a display
    os.environ["SDL_VIDEODRIVER"] = ""
    pygame.display.init()
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")

    terminated = False
    
    # Use a simple human-like policy for demonstration
    key_map = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while not terminated:
        movement_action = 0 # No-op
        space_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement_action = 3
        if keys[pygame.K_RIGHT]:
            movement_action = 4
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        action = [movement_action, space_action, 0] # Shift is not used

        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Run at 30 FPS

    pygame.quit()
    print(f"Game Over. Final Score: {info['score']}")