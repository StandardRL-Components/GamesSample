
# Generated: 2025-08-28T06:10:32.958766
# Source Brief: brief_02857.md
# Brief Index: 2857

        
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
        "Controls: ←→ to move the paddle. Break all the blocks to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker where risk-taking is rewarded. "
        "Clear the screen of all 100 blocks, but be careful not to lose all 3 of your balls."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAYFIELD_PADDING = 10
    MAX_STEPS = 2000
    
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_WALL = (80, 80, 100)
    COLOR_PADDLE = (220, 220, 255)
    COLOR_PADDLE_GLOW = (100, 100, 200)
    COLOR_BALL = (0, 255, 255)
    COLOR_BALL_GLOW = (0, 150, 150)
    COLOR_TEXT = (255, 255, 255)
    BLOCK_COLORS = [
        (255, 0, 128), (255, 128, 0), (255, 255, 0), (0, 255, 0), (0, 128, 255)
    ]

    # Game Object Properties
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 10
    BALL_RADIUS = 7
    BALL_SPEED = 6
    INITIAL_BALLS = 3
    
    BLOCK_ROWS = 10
    BLOCK_COLS = 10
    BLOCK_AREA_HEIGHT = 150
    BLOCK_SPACING = 4

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Playfield rect
        self.playfield = pygame.Rect(
            self.PLAYFIELD_PADDING,
            self.PLAYFIELD_PADDING,
            self.SCREEN_WIDTH - 2 * self.PLAYFIELD_PADDING,
            self.SCREEN_HEIGHT - 2 * self.PLAYFIELD_PADDING
        )
        
        # Block dimensions based on playfield
        self.BLOCK_WIDTH = (self.playfield.width - (self.BLOCK_COLS + 1) * self.BLOCK_SPACING) / self.BLOCK_COLS
        self.BLOCK_HEIGHT = (self.BLOCK_AREA_HEIGHT - (self.BLOCK_ROWS - 1) * self.BLOCK_SPACING) / self.BLOCK_ROWS

        # Initialize state variables to be set in reset()
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = None
        self.particles = None
        self.balls_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.blocks_broken_this_hit = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = self.INITIAL_BALLS
        
        self.paddle = pygame.Rect(
            self.playfield.centerx - self.PADDLE_WIDTH / 2,
            self.playfield.bottom - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self._reset_ball()
        
        self.blocks = []
        for row in range(self.BLOCK_ROWS):
            for col in range(self.BLOCK_COLS):
                x = self.playfield.left + self.BLOCK_SPACING + col * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = self.playfield.top + self.BLOCK_SPACING + row * (self.BLOCK_HEIGHT + self.BLOCK_SPACING) + 40 # Offset from top
                block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                color_index = (row // 2) % len(self.BLOCK_COLORS)
                self.blocks.append({"rect": block_rect, "color": self.BLOCK_COLORS[color_index]})

        self.particles = []
        self.blocks_broken_this_hit = 0
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1)
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Upwards angle
        self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_SPEED
        self.blocks_broken_this_hit = 0

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.2 # Continuous penalty for time passing

        # Unpack factorized action
        movement = action[0]
        
        # --- Update game logic ---
        self._handle_input(movement)
        self._update_ball()
        reward += self._handle_collisions()
        self._update_particles()
        
        self.steps += 1
        self.score += reward
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if not self.blocks: # Win condition
                reward += 100
                self.score += 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to playfield
        self.paddle.left = max(self.playfield.left, self.paddle.left)
        self.paddle.right = min(self.playfield.right, self.paddle.right)

    def _update_ball(self):
        self.ball_pos += self.ball_vel

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS <= self.playfield.left:
            self.ball_pos.x = self.playfield.left + self.BALL_RADIUS
            self.ball_vel.x *= -1
        if self.ball_pos.x + self.BALL_RADIUS >= self.playfield.right:
            self.ball_pos.x = self.playfield.right - self.BALL_RADIUS
            self.ball_vel.x *= -1
        if self.ball_pos.y - self.BALL_RADIUS <= self.playfield.top:
            self.ball_pos.y = self.playfield.top + self.BALL_RADIUS
            self.ball_vel.y *= -1
        
        # Bottom wall (lose ball)
        if self.ball_pos.y + self.BALL_RADIUS >= self.playfield.bottom:
            self.balls_left -= 1
            reward -= 50
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True
            return reward

        # Paddle collision
        if ball_rect.colliderect(self.paddle):
            # Sound effect placeholder: # sfx_paddle_hit
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
            self.ball_vel.y *= -1
            
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += offset * 2.0
            self.ball_vel = self.ball_vel.normalize() * self.BALL_SPEED
            
            reward += 0.1
            if self.blocks_broken_this_hit >= 3:
                reward += 5 # Combo bonus
            self.blocks_broken_this_hit = 0

        # Block collisions
        for i, block_data in enumerate(self.blocks):
            block = block_data["rect"]
            if ball_rect.colliderect(block):
                # Sound effect placeholder: # sfx_block_break
                self._create_particles(block.center, block_data["color"])
                
                # Determine collision side to correctly reflect ball
                overlap = ball_rect.clip(block)
                if overlap.width < overlap.height:
                    self.ball_vel.x *= -1
                else:
                    self.ball_vel.y *= -1

                self.blocks.pop(i)
                reward += 1
                self.blocks_broken_this_hit += 1
                break # Only handle one block collision per frame

        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(10, 25)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _check_termination(self):
        return self.balls_left <= 0 or not self.blocks or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw playfield walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, self.playfield, 2)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 25))
            color = (*p['color'], alpha)
            size = int(self.BALL_RADIUS * 0.5 * (p['lifespan'] / 25))
            if size > 0:
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(p['pos'].x - size), int(p['pos'].y - size)), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw blocks
        for block_data in self.blocks:
            pygame.draw.rect(self.screen, block_data["color"], block_data["rect"])
            # Add a subtle 3D effect
            pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in block_data["color"]), block_data["rect"].inflate(-6, -6), 2)

        # Draw paddle with glow
        glow_rect = self.paddle.inflate(6, 6)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PADDLE_GLOW, 100), glow_surf.get_rect(), border_radius=5)
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball with glow
        if self.balls_left > 0 or self.game_over:
            # Glow
            glow_radius = int(self.BALL_RADIUS * 2)
            glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*self.COLOR_BALL_GLOW, 80))
            self.screen.blit(glow_surf, (int(self.ball_pos.x - glow_radius), int(self.ball_pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
            # Ball
            pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.playfield.left + 10, self.playfield.top + 5))
        
        # Balls left
        for i in range(self.balls_left):
            x = self.playfield.right - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            y = self.playfield.top + 10 + self.BALL_RADIUS
            pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

        # Game Over / Win message
        if self.game_over:
            if not self.blocks:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_game_over.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
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

# Example usage for testing
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # To run the game with manual controls
    pygame.display.set_caption(env.game_description)
    real_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    action = env.action_space.sample()
    action[0] = 0 # No-op
    action[1] = 0
    action[2] = 0
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        action[0] = 0 # Default to no-op
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Blit the env's internal screen to the real screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        real_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Limit to 30 FPS for playability

    env.close()