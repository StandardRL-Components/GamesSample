
# Generated: 2025-08-28T07:06:53.417545
# Source Brief: brief_03135.md
# Brief Index: 3135

        
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
        "Controls: ← to move the paddle left, → to move right. Deflect the ball to break all bricks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade brick-breaker. Survive 30 waves of increasing difficulty by deflecting the ball with your paddle. Don't let the ball hit the bottom!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # For auto_advance=True, this is the target update rate
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_WALL = (80, 80, 100)
        self.BRICK_COLORS = [
            (50, 205, 50),   # Green (1 HP)
            (255, 215, 0),   # Yellow (2 HP)
            (220, 20, 60),   # Red (3 HP)
        ]
        self.PARTICLE_COLORS = [(255, 165, 0), (255, 140, 0), (255, 69, 0)]
        self.COLOR_TEXT = (240, 240, 240)

        # Game parameters
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 16
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 6
        self.MAX_WAVES = 30
        self.MAX_STEPS = 10000
        self.WALL_THICKNESS = 10
        
        # Play area (leaving space for walls and UI)
        self.PLAY_AREA_X = self.WALL_THICKNESS
        self.PLAY_AREA_Y = 50
        self.PLAY_AREA_WIDTH = self.WIDTH - 2 * self.WALL_THICKNESS
        
        self.BRICK_ROWS = 8
        self.BRICK_COLS = 14
        self.BRICK_WIDTH = self.PLAY_AREA_WIDTH / self.BRICK_COLS
        self.BRICK_HEIGHT = 20
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        
        # Etc...        
        self.paddle = None
        self.balls = []
        self.bricks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.current_wave = 0
        self.game_over = False
        self.game_won = False
        self.reward_this_step = 0.0
        self.np_random = None
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.current_wave = 0
        self.game_over = False
        self.game_won = False
        
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT - 20
        paddle_x = (self.WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        
        self.balls.clear()
        self.particles.clear()
        
        self._next_wave()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _next_wave(self):
        self.current_wave += 1
        
        # Spawn one ball
        self.balls.clear()
        ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 5)
        
        base_speed = 4.0
        wave_speed_bonus = (self.current_wave - 1) * 0.05
        initial_speed = base_speed + wave_speed_bonus
        
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * initial_speed
        
        self.balls.append({"pos": ball_pos, "vel": vel, "rect": pygame.Rect(0,0,0,0)})
        
        self._generate_bricks()
        
    def _generate_bricks(self):
        self.bricks.clear()
        for r in range(self.BRICK_ROWS):
            for c in range(self.BRICK_COLS):
                spawn_chance = 0.3 + (self.current_wave / self.MAX_WAVES) * 0.5 + (1 - abs(c - self.BRICK_COLS/2)/(self.BRICK_COLS/2)) * 0.2
                if self.np_random.random() < spawn_chance:
                    hp = 1
                    brick_x = self.PLAY_AREA_X + c * self.BRICK_WIDTH
                    brick_y = self.PLAY_AREA_Y + r * self.BRICK_HEIGHT
                    rect = pygame.Rect(brick_x, brick_y, self.BRICK_WIDTH - 2, self.BRICK_HEIGHT - 2)
                    self.bricks.append({"rect": rect, "hp": hp})
    
    def step(self, action):
        if self.game_over or self.game_won:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        self.reward_this_step = -0.02
        
        # Unpack factorized action
        movement = action[0]
        
        # Update game logic
        self._handle_input(movement)
        self._update_game_state()
        
        self.steps += 1
        
        reward = self.reward_this_step
        terminated = self.game_over or self.steps >= self.MAX_STEPS or self.game_won
        
        if self.game_over:
            reward = -100.0
        if self.game_won:
            reward = 100.0
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = max(self.PLAY_AREA_X, self.paddle.x)
        self.paddle.x = min(self.WIDTH - self.PLAY_AREA_X - self.PADDLE_WIDTH, self.paddle.x)
        
    def _update_game_state(self):
        self._update_balls()
        self._update_particles()
        
        if not self.bricks and not self.game_over:
            self.score += 10 * self.current_wave
            self.reward_this_step += 1.0 # Wave clear reward
            if self.current_wave >= self.MAX_WAVES:
                self.game_won = True
            else:
                # sfx: wave clear
                self._next_wave()
                
    def _update_balls(self):
        for ball in self.balls[:]:
            ball["pos"] += ball["vel"]
            ball["rect"] = pygame.Rect(
                ball["pos"].x - self.BALL_RADIUS, 
                ball["pos"].y - self.BALL_RADIUS, 
                self.BALL_RADIUS * 2, 
                self.BALL_RADIUS * 2
            )
            
            # Wall collisions
            if ball["rect"].left <= self.PLAY_AREA_X or ball["rect"].right >= self.WIDTH - self.PLAY_AREA_X:
                ball["vel"].x *= -1
                ball["rect"].clamp_ip(pygame.Rect(self.PLAY_AREA_X, 0, self.PLAY_AREA_WIDTH, self.HEIGHT))
                ball["pos"].x = ball["rect"].centerx
                # sfx: bounce
            if ball["rect"].top <= self.PLAY_AREA_Y:
                ball["vel"].y *= -1
                ball["rect"].top = self.PLAY_AREA_Y
                ball["pos"].y = ball["rect"].centery
                # sfx: bounce

            # Bottom wall collision (Game Over)
            if ball["rect"].bottom >= self.HEIGHT:
                self.game_over = True
                # sfx: game over
                return
                
            # Paddle collision
            if ball["rect"].colliderect(self.paddle) and ball["vel"].y > 0:
                ball["vel"].y *= -1
                offset = ball["rect"].centerx - self.paddle.centerx
                normalized_offset = offset / (self.PADDLE_WIDTH / 2)
                speed = ball["vel"].length()
                ball["vel"].x += normalized_offset * 2.0
                ball["vel"] = ball["vel"].normalize() * speed
                ball["rect"].bottom = self.paddle.top
                ball["pos"].y = ball["rect"].centery
                # sfx: paddle hit
                
            # Brick collision
            for brick in self.bricks[:]:
                if ball["rect"].colliderect(brick["rect"]):
                    # sfx: brick hit
                    self.bricks.remove(brick)
                    self.score += 1
                    self.reward_this_step += 0.1 # Brick destroyed reward
                    
                    for _ in range(5):
                        self._spawn_particle(brick["rect"].center)
                    
                    # Determine collision side for accurate bounce
                    hit_rect = ball["rect"].clip(brick["rect"])
                    if hit_rect.width > hit_rect.height:
                        ball["vel"].y *= -1 # Vertical collision
                    else:
                        ball["vel"].x *= -1 # Horizontal collision
                    break
    
    def _spawn_particle(self, pos):
        vel = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2))
        lifetime = self.np_random.integers(10, 20)
        color = random.choice(self.PARTICLE_COLORS)
        self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "lifetime": lifetime, "color": color})
        
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.9
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, self.PLAY_AREA_Y - self.WALL_THICKNESS, self.WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.HEIGHT))
        
        for brick in self.bricks:
            color_index = min(len(self.BRICK_COLORS) - 1, brick["hp"] - 1)
            pygame.draw.rect(self.screen, self.BRICK_COLORS[color_index], brick["rect"], border_radius=3)
            
        for p in self.particles:
            size = max(0, p["lifetime"] * 0.3)
            pygame.draw.circle(self.screen, p["color"], p["pos"], size)

        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)
        
        for ball in self.balls:
            pygame.gfxdraw.aacircle(self.screen, int(ball["pos"].x), int(ball["pos"].y), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, int(ball["pos"].x), int(ball["pos"].y), self.BALL_RADIUS, self.COLOR_BALL)
            
        if self.game_over:
            text = self.font_game_over.render("GAME OVER", True, (255, 0, 0))
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text, text_rect)
        elif self.game_won:
            text = self.font_game_over.render("YOU WIN!", True, (0, 255, 0))
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text, text_rect)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))
        
        wave_text = self.font_ui.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        wave_rect = wave_text.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(wave_text, wave_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset to get initial observation
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {obs.shape}, expected {(self.HEIGHT, self.WIDTH, 3)}"
        assert obs.dtype == np.uint8
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

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Grid Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    
    while not terminated:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()

    env.close()