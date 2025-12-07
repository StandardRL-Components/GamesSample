
# Generated: 2025-08-27T19:16:47.767652
# Source Brief: brief_02106.md
# Brief Index: 2106

        
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
        "Controls: ←→ to move the paddle. Break all the blocks to win. Don't let the ball hit the bottom!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, retro block breaker. Use the paddle to bounce the ball, destroy blocks, and chase high scores through combos."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 10000

    # Colors
    COLOR_BG = (15, 15, 30)
    COLOR_GRID = (30, 30, 50)
    COLOR_WALL = (100, 100, 120)
    COLOR_PADDLE = (240, 240, 240)
    COLOR_BALL = (255, 255, 0)
    COLOR_TEXT = (255, 255, 255)
    BLOCK_COLORS = [
        (255, 0, 128),  # Magenta
        (255, 100, 0),  # Orange
        (0, 200, 255),  # Cyan
        (0, 255, 0),    # Green
        (160, 32, 240)  # Purple
    ]

    # Game Object Sizes
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 12
    BALL_RADIUS = 7
    WALL_THICKNESS = 10
    
    # Block Grid
    BLOCK_ROWS = 5
    BLOCK_COLS = 20
    BLOCK_WIDTH = 30
    BLOCK_HEIGHT = 15
    BLOCK_H_SPACING = 2
    BLOCK_V_SPACING = 2
    BLOCK_GRID_TOP_MARGIN = 50

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
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables to avoid AttributeError
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.paddle = None
        self.ball_pos = np.zeros(2, dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.blocks = []
        self.particles = []
        self.combo_counter = 0
        self.combo_steps_since_last_hit = 0
        self.blocks_destroyed_count = 0
        self.speed_boosts_applied = 0
        self.win_condition_met = False
        self._rng = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        self.lives = 3
        
        # Paddle
        paddle_y = self.SCREEN_HEIGHT - 30
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        # Ball
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1], dtype=np.float32)
        initial_angle = self._rng.uniform(math.pi * 0.25, math.pi * 0.75)
        initial_speed = 2.0
        self.ball_vel = np.array([math.cos(initial_angle) * initial_speed, -math.sin(initial_angle) * initial_speed])
        
        # Blocks
        self.blocks = []
        grid_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_H_SPACING)
        start_x = (self.SCREEN_WIDTH - grid_width) / 2 + self.BLOCK_H_SPACING
        for r in range(self.BLOCK_ROWS):
            for c in range(self.BLOCK_COLS):
                x = start_x + c * (self.BLOCK_WIDTH + self.BLOCK_H_SPACING)
                y = self.BLOCK_GRID_TOP_MARGIN + r * (self.BLOCK_HEIGHT + self.BLOCK_V_SPACING)
                color = self.BLOCK_COLORS[r % len(self.BLOCK_COLORS)]
                self.blocks.append({"rect": pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT), "color": color, "active": True})

        # Particles & Effects
        self.particles = []
        self.combo_counter = 0
        self.combo_steps_since_last_hit = 999
        self.blocks_destroyed_count = 0
        self.speed_boosts_applied = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.02  # Time penalty to encourage efficiency
        self.steps += 1
        self.combo_steps_since_last_hit += 1

        # 1. Handle Input
        movement = action[0]
        paddle_speed = 10
        if movement == 3:  # Left
            self.paddle.x -= paddle_speed
        elif movement == 4:  # Right
            self.paddle.x += paddle_speed
        
        self.paddle.x = np.clip(self.paddle.x, self.WALL_THICKNESS, self.SCREEN_WIDTH - self.PADDLE_WIDTH - self.WALL_THICKNESS)

        # 2. Update Ball Position
        self.ball_pos += self.ball_vel
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # 3. Collision Detection
        # Walls
        if ball_rect.left <= self.WALL_THICKNESS:
            ball_rect.left = self.WALL_THICKNESS
            self.ball_vel[0] *= -1
        if ball_rect.right >= self.SCREEN_WIDTH - self.WALL_THICKNESS:
            ball_rect.right = self.SCREEN_WIDTH - self.WALL_THICKNESS
            self.ball_vel[0] *= -1
        if ball_rect.top <= self.WALL_THICKNESS:
            ball_rect.top = self.WALL_THICKNESS
            self.ball_vel[1] *= -1
        
        # Bottom (lose life)
        if ball_rect.top > self.SCREEN_HEIGHT:
            self.lives -= 1
            if self.lives > 0:
                # Reset ball position
                self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 5], dtype=np.float32)
                initial_angle = self._rng.uniform(math.pi * 0.4, math.pi * 0.6)
                self.ball_vel = np.array([math.cos(initial_angle) * 2.0, -math.sin(initial_angle) * 2.0])
            else:
                self.game_over = True
                reward -= 100

        # Paddle
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # # PADDLE_HIT_SOUND
            self.ball_vel[1] *= -1
            ball_rect.bottom = self.paddle.top
            
            # Paddle hit physics and reward
            hit_pos = ball_rect.centerx - self.paddle.centerx
            influence = hit_pos / (self.PADDLE_WIDTH / 2.0)
            self.ball_vel[0] = influence * 3.0 # Max horizontal speed influence
            
            if abs(hit_pos) <= self.PADDLE_WIDTH * 0.1: # Center 20%
                reward += 0.1
            elif abs(hit_pos) >= self.PADDLE_WIDTH * 0.4: # Outer 10% each side
                reward -= 0.2

            self.combo_counter = 0 # Reset combo on paddle hit
            self._normalize_ball_velocity()

        # Blocks
        for block in self.blocks:
            if block["active"] and ball_rect.colliderect(block["rect"]):
                # # BLOCK_BREAK_SOUND
                block["active"] = False
                reward += 1
                self.score += 10
                self.blocks_destroyed_count += 1
                
                self._create_particles(block["rect"].center, block["color"])
                
                # Combo logic
                if self.combo_steps_since_last_hit <= 5:
                    self.combo_counter += 1
                    reward += 5
                    self.score += self.combo_counter * 10
                else:
                    self.combo_counter = 1
                self.combo_steps_since_last_hit = 0

                # Ball reflection logic
                self.ball_vel[1] *= -1 # Simple vertical reflection
                
                # Difficulty scaling
                if self.blocks_destroyed_count > 0 and self.blocks_destroyed_count % 10 == 0:
                    if self.blocks_destroyed_count // 10 > self.speed_boosts_applied:
                        self.speed_boosts_applied += 1
                        self._increase_ball_speed(0.05)
                break # Only one block per frame
        
        self.ball_pos = np.array([ball_rect.centerx, ball_rect.centery], dtype=np.float32)
        
        # 4. Update Particles
        self._update_particles()
        
        # 5. Check Termination Conditions
        if self.blocks_destroyed_count == self.BLOCK_ROWS * self.BLOCK_COLS:
            self.win_condition_met = True
            self.game_over = True
            reward += 100
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _normalize_ball_velocity(self):
        current_speed = np.linalg.norm(self.ball_vel)
        base_speed = 2.0 + self.speed_boosts_applied * 0.05
        if current_speed > 0:
            self.ball_vel = (self.ball_vel / current_speed) * base_speed
        
        # Clamp vertical speed to prevent stalemates
        if abs(self.ball_vel[1]) < 0.5:
            self.ball_vel[1] = 0.5 * np.sign(self.ball_vel[1])

    def _increase_ball_speed(self, amount):
        current_speed = np.linalg.norm(self.ball_vel)
        if current_speed > 0:
            new_speed = current_speed + amount
            self.ball_vel = (self.ball_vel / current_speed) * new_speed

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self._rng.uniform(0, 2 * math.pi)
            speed = self._rng.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self._rng.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Background grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        
        # Blocks
        for block in self.blocks:
            if block["active"]:
                pygame.gfxdraw.box(self.screen, block["rect"], block["color"])
                brighter_color = tuple(min(255, c + 30) for c in block["color"])
                pygame.draw.rect(self.screen, brighter_color, block["rect"], 1)

        # Particles
        for p in self.particles:
            size = int(p['lifespan'] / 5)
            if size > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, p['color'])

        # Ball
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        glow_radius = int(self.BALL_RADIUS * 2.5)
        glow_color = (*self.COLOR_BALL, 50)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, glow_radius, glow_radius, glow_radius, glow_color)
        self.screen.blit(s, (ball_pos_int[0] - glow_radius, ball_pos_int[1] - glow_radius))
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        # Paddle
        pygame.gfxdraw.box(self.screen, self.paddle, self.COLOR_PADDLE)
        pygame.draw.rect(self.screen, (255, 255, 255), self.paddle, 1)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, self.WALL_THICKNESS + 5))
        
        # Lives
        lives_text = self.font_medium.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 160, self.WALL_THICKNESS + 5))
        for i in range(self.lives):
            pygame.gfxdraw.filled_circle(self.screen, self.SCREEN_WIDTH - 60 + i * 20, self.WALL_THICKNESS + 15, self.BALL_RADIUS - 2, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, self.SCREEN_WIDTH - 60 + i * 20, self.WALL_THICKNESS + 15, self.BALL_RADIUS - 2, self.COLOR_BALL)
            
        # Combo
        if self.combo_counter > 1 and self.combo_steps_since_last_hit < 30:
            combo_text = self.font_large.render(f"x{self.combo_counter} COMBO!", True, self.COLOR_BALL)
            text_rect = combo_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(combo_text, text_rect)

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win_condition_met else "GAME OVER"
            color = (0, 255, 0) if self.win_condition_met else (255, 0, 0)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_left": self.BLOCK_ROWS * self.BLOCK_COLS - self.blocks_destroyed_count,
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Movement
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Wait 3 seconds before closing
            
        clock.tick(30) # Run at 30 FPS

    env.close()