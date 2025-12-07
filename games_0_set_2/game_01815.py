
# Generated: 2025-08-28T02:47:06.346957
# Source Brief: brief_01815.md
# Brief Index: 1815

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade block breaker. Clear all blocks on the screen by bouncing a ball with your paddle. Clear all 3 stages to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.BALL_BASE_SPEED = 7
        self.MAX_LIVES = 3
        self.STAGE_TIME_SECONDS = 60
        self.BOUNDARY_THICKNESS = 10

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_WALL = (100, 100, 120)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_PADDLE_GLOW = (200, 200, 255)
        self.COLOR_BALL_GLOW = (200, 200, 255)
        self.BLOCK_DEFINITIONS = {
            "red": {"color": (255, 70, 70), "points": 1, "glow": (255, 150, 150)},
            "green": {"color": (70, 255, 70), "points": 2, "glow": (150, 255, 150)},
            "blue": {"color": (70, 70, 255), "points": 3, "glow": (150, 150, 255)},
        }

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.score = None
        self.lives = None
        self.timer = None
        self.stage = None
        self.game_over = None
        self.game_won = None
        self.steps = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.lives = self.MAX_LIVES
        self.stage = 1
        self.game_over = False
        self.game_won = False
        self.steps = 0
        self.particles = []
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        self.timer = self.STAGE_TIME_SECONDS * self.FPS
        
        # Paddle
        paddle_y = self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        # Ball
        self.ball_launched = False
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)

        # Blocks
        self.blocks = []
        block_width, block_height = 50, 20
        rows, cols = 0, 0
        
        if self.stage == 1:
            rows, cols = 4, 10
            colors = ["red", "red", "green", "green"]
        elif self.stage == 2:
            rows, cols = 5, 10
            colors = ["red", "green", "green", "blue", "blue"]
        elif self.stage == 3:
            rows, cols = 6, 10
            colors = ["red", "red", "green", "green", "blue", "blue"]

        x_offset = (self.SCREEN_WIDTH - cols * (block_width + 4) + 4) / 2
        y_offset = 50

        for r in range(rows):
            for c in range(cols):
                if self.stage == 3 and (c < 2 or c > 7): continue # Create a pattern for stage 3
                block_type = colors[r % len(colors)]
                block_def = self.BLOCK_DEFINITIONS[block_type]
                self.blocks.append({
                    "rect": pygame.Rect(x_offset + c * (block_width + 4), y_offset + r * (block_height + 4), block_width, block_height),
                    "type": block_type,
                    "color": block_def["color"],
                    "points": block_def["points"],
                    "glow": block_def["glow"]
                })

    def step(self, action):
        reward = -0.01  # Time penalty
        self.steps += 1
        self.timer -= 1

        # 1. Handle Input
        movement = action[0]
        space_pressed = action[1] == 1

        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.left = max(self.BOUNDARY_THICKNESS, self.paddle.left)
        self.paddle.right = min(self.SCREEN_WIDTH - self.BOUNDARY_THICKNESS, self.paddle.right)

        if space_pressed and not self.ball_launched:
            self.ball_launched = True
            ball_speed = self.BALL_BASE_SPEED + (self.stage - 1) * 0.5
            angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4)
            self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * ball_speed
            # sfx: launch_ball

        # 2. Update Game State
        if not self.ball_launched:
            self.ball_pos.x = self.paddle.centerx
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
        else:
            self.ball_pos += self.ball_vel
            
            # Ball-wall collision
            if self.ball_pos.x - self.BALL_RADIUS <= self.BOUNDARY_THICKNESS:
                self.ball_pos.x = self.BOUNDARY_THICKNESS + self.BALL_RADIUS
                self.ball_vel.x *= -1
                # sfx: wall_bounce
            if self.ball_pos.x + self.BALL_RADIUS >= self.SCREEN_WIDTH - self.BOUNDARY_THICKNESS:
                self.ball_pos.x = self.SCREEN_WIDTH - self.BOUNDARY_THICKNESS - self.BALL_RADIUS
                self.ball_vel.x *= -1
                # sfx: wall_bounce
            if self.ball_pos.y - self.BALL_RADIUS <= self.BOUNDARY_THICKNESS:
                self.ball_pos.y = self.BOUNDARY_THICKNESS + self.BALL_RADIUS
                self.ball_vel.y *= -1
                # sfx: wall_bounce

            # Ball-paddle collision
            ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
            if ball_rect.colliderect(self.paddle) and self.ball_vel.y > 0:
                reward += 0.1
                self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
                
                # Dynamic bounce angle
                offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                bounce_angle = math.radians(offset * 60) # Max 60 degree influence
                current_speed = self.ball_vel.length()
                self.ball_vel.x = current_speed * math.sin(bounce_angle)
                self.ball_vel.y = -current_speed * math.cos(bounce_angle)
                self.ball_vel.y = -abs(self.ball_vel.y) # Ensure it always goes up
                # sfx: paddle_bounce

            # Ball-block collision
            for block in self.blocks[:]:
                if ball_rect.colliderect(block["rect"]):
                    reward += block["points"]
                    self.score += block["points"] * 10
                    self._create_particles(block["rect"].center, block["color"])
                    self.blocks.remove(block)
                    # sfx: block_break
                    
                    # Simple collision response
                    self.ball_vel.y *= -1
                    break

            # Ball out of bounds
            if self.ball_pos.y > self.SCREEN_HEIGHT:
                reward -= 10
                self.lives -= 1
                self.ball_launched = False
                if self.lives <= 0:
                    self.game_over = True
                # sfx: lose_life

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
        
        # 3. Check for Termination / Progression
        terminated = False
        if not self.game_over:
            if not self.blocks: # Stage clear
                reward += 50
                self.stage += 1
                if self.stage > 3:
                    self.game_won = True
                    reward += 100
                    terminated = True
                else:
                    self._setup_stage()
                    # sfx: stage_clear
            
            if self.timer <= 0:
                self.game_over = True

        if self.game_over or self.game_won:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'lifespan': self.np_random.integers(10, 20),
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Draw boundaries
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.BOUNDARY_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.BOUNDARY_THICKNESS, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - self.BOUNDARY_THICKNESS, 0, self.BOUNDARY_THICKNESS, self.SCREEN_HEIGHT))

        # Draw blocks with glow
        for block in self.blocks:
            glow_rect = block["rect"].inflate(4, 4)
            shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, (*block["glow"], 60), (0, 0, *glow_rect.size), border_radius=4)
            self.screen.blit(shape_surf, glow_rect.topleft)
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['lifespan'] / 20))
            color_with_alpha = (*p['color'], alpha)
            shape_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(shape_surf, color_with_alpha, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(shape_surf, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

        # Draw paddle with glow
        glow_rect = self.paddle.inflate(6, 6)
        shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, (*self.COLOR_PADDLE_GLOW, 80), (0, 0, *glow_rect.size), border_radius=6)
        self.screen.blit(shape_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Draw ball with glow
        ball_pos_int = (int(self.ball_pos.x), int(self.ball_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 3, (*self.COLOR_BALL_GLOW, 80))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Draw UI
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.BOUNDARY_THICKNESS + 10, self.SCREEN_HEIGHT - 30))

        timer_text = self.font_ui.render(f"TIME: {self.timer // self.FPS:02d}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH // 2 - timer_text.get_width() // 2, self.SCREEN_HEIGHT - 30))

        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - self.BOUNDARY_THICKNESS - 10, self.SCREEN_HEIGHT - 30))
        
        stage_text = self.font_ui.render(f"STAGE: {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.SCREEN_WIDTH // 2 - stage_text.get_width() // 2, 15))


        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER"
            color = self.BLOCK_DEFINITIONS["red"]["color"]
            msg_render = self.font_game_over.render(msg, True, color)
            self.screen.blit(msg_render, (self.SCREEN_WIDTH // 2 - msg_render.get_width() // 2, self.SCREEN_HEIGHT // 2 - msg_render.get_height() // 2))
        elif self.game_won:
            msg = "YOU WIN!"
            color = self.BLOCK_DEFINITIONS["green"]["color"]
            msg_render = self.font_game_over.render(msg, True, color)
            self.screen.blit(msg_render, (self.SCREEN_WIDTH // 2 - msg_render.get_width() // 2, self.SCREEN_HEIGHT // 2 - msg_render.get_height() // 2))

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
            "game_over": self.game_over,
            "game_won": self.game_won
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        # Map keyboard inputs to the action space
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        clock.tick(env.FPS)
        
    env.close()