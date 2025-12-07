
# Generated: 2025-08-27T13:11:44.661193
# Source Brief: brief_00292.md
# Brief Index: 292

        
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
        "Controls: ↑/↓ to move the paddle. Press space to fire a powerful, limited special shot."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro grid-based Pong game. Hit the ball to score, but miss and you lose a life. Use your special shots wisely for bonus points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 15
        self.CELL_SIZE = 25
        self.GAME_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.GAME_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE
        self.OFFSET_X = (self.SCREEN_WIDTH - self.GAME_WIDTH) // 2
        self.OFFSET_Y = (self.SCREEN_HEIGHT - self.GAME_HEIGHT) // 2

        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 1, 4
        self.BALL_SIZE = 1
        self.INITIAL_BALL_SPEED = 1.0
        self.MAX_STEPS = 1500 # Increased for longer rallies
        self.WIN_SCORE = 7
        self.MAX_LIVES = 5
        self.INITIAL_SPECIAL_SHOTS = 2

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (100, 200, 255)
        self.COLOR_SPECIAL_SHOT = (255, 255, 0)
        self.COLOR_TEXT = (50, 255, 50)
        self.COLOR_MISS_FLASH = (180, 20, 20)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_big = pygame.font.SysFont("Consolas", 32, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 20)
        except pygame.error:
            self.font_big = pygame.font.Font(None, 40)
            self.font_small = pygame.font.Font(None, 24)

        # --- State Variables ---
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = None
        self.score = 0
        self.lives = 0
        self.special_shots = 0
        self.score_for_next_speedup = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.special_shot_effect = None
        self.particles = []
        self.screen_flash_timer = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.paddle = pygame.Rect(
            self.GRID_WIDTH - self.PADDLE_WIDTH - 1,
            (self.GRID_HEIGHT - self.PADDLE_HEIGHT) // 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.ball_pos = np.array([self.GRID_WIDTH / 2, self.GRID_HEIGHT / 2], dtype=float)
        
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        if self.np_random.choice([True, False]):
            angle += math.pi
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)])
        
        self.ball_speed = self.INITIAL_BALL_SPEED
        self.score = 0
        self.lives = self.MAX_LIVES
        self.special_shots = self.INITIAL_SPECIAL_SHOTS
        self.score_for_next_speedup = 2
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.special_shot_effect = None
        self.particles = []
        self.screen_flash_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1

        # --- Handle Actions ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Paddle Movement
        if movement == 1:  # Up
            self.paddle.y -= 1
        elif movement == 2:  # Down
            self.paddle.y += 1
        self.paddle.y = np.clip(self.paddle.y, 0, self.GRID_HEIGHT - self.PADDLE_HEIGHT)

        # Special Shot (triggered on press, not hold)
        if space_held and not self.prev_space_held and self.special_shots > 0:
            self.special_shots -= 1
            self.special_shot_effect = {
                "center": self.paddle.center,
                "radius": 0,
                "max_radius": 4,
                "duration": 15,
                "hit": False
            }
            # sfx: special_shot_fire

        self.prev_space_held = space_held

        # --- Update Game State ---
        self._update_ball()
        self._update_particles()
        if self.special_shot_effect:
            self._update_special_shot()

        # --- Calculate Reward ---
        # This is calculated inside _update_ball based on events
        reward += self.step_reward
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            elif self.lives <= 0:
                reward -= 100 # Loss penalty

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_ball(self):
        self.step_reward = 0
        self.ball_pos += self.ball_vel * self.ball_speed

        bx, by = self.ball_pos
        b_rect = pygame.Rect(int(bx), int(by), 1, 1)

        # Wall collisions
        if by < 0:
            self.ball_pos[1] = 0
            self.ball_vel[1] *= -1
            self._create_particles(self.ball_pos, self.COLOR_BALL, 5) # sfx: wall_bounce
        elif by > self.GRID_HEIGHT - 1:
            self.ball_pos[1] = self.GRID_HEIGHT - 1
            self.ball_vel[1] *= -1
            self._create_particles(self.ball_pos, self.COLOR_BALL, 5) # sfx: wall_bounce

        if bx < 0:
            self.ball_pos[0] = 0
            self.ball_vel[0] *= -1
            self._create_particles(self.ball_pos, self.COLOR_BALL, 5) # sfx: wall_bounce

        # Paddle collision
        if self.ball_vel[0] > 0 and self.paddle.colliderect(b_rect):
            self.ball_pos[0] = self.paddle.left - 1
            self.ball_vel[0] *= -1
            
            # Add spin based on where it hits the paddle
            hit_pos_norm = (self.paddle.centery - by) / (self.PADDLE_HEIGHT / 2)
            self.ball_vel[1] -= hit_pos_norm * 0.5
            self.ball_vel /= np.linalg.norm(self.ball_vel) # Normalize to maintain speed

            self.score += 1
            self.step_reward += 1
            self._create_particles(self.ball_pos, self.COLOR_PADDLE, 15) # sfx: paddle_hit

            if self.score >= self.score_for_next_speedup:
                self.ball_speed += 0.2
                self.score_for_next_speedup += 2
                # sfx: speed_up

        # Special shot collision
        if self.special_shot_effect and not self.special_shot_effect["hit"]:
            dist = np.linalg.norm(self.ball_pos - np.array(self.special_shot_effect["center"]))
            if dist <= self.special_shot_effect["radius"]:
                self.special_shot_effect["hit"] = True
                self.score += 3
                self.step_reward += 3
                self.ball_vel *= -1.2 # Knockback effect
                # sfx: special_hit
        
        # Miss
        if bx > self.GRID_WIDTH:
            self.lives -= 1
            self.step_reward -= 1
            self.screen_flash_timer = 10
            self._reset_ball_position()
            # sfx: miss
    
    def _reset_ball_position(self):
        self.ball_pos = np.array([self.GRID_WIDTH / 2, self.GRID_HEIGHT / 2], dtype=float)
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)])

    def _update_special_shot(self):
        self.special_shot_effect["duration"] -= 1
        if self.special_shot_effect["hit"]:
            self.special_shot_effect["radius"] += 0.5 # Visual feedback on hit
        else:
            self.special_shot_effect["radius"] += self.special_shot_effect["max_radius"] / 15
        
        if self.special_shot_effect["duration"] <= 0:
            self.special_shot_effect = None

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
    
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.1, 0.5)
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle), math.sin(angle)]) * speed,
                "life": self.np_random.integers(10, 20),
                "color": color
            })

    def _check_termination(self):
        return self.lives <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {"score": self.score, "lives": self.lives, "steps": self.steps}

    def _grid_to_pixels(self, grid_pos):
        x = self.OFFSET_X + grid_pos[0] * self.CELL_SIZE
        y = self.OFFSET_Y + grid_pos[1] * self.CELL_SIZE
        return int(x), int(y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_WIDTH + 1):
            start = self._grid_to_pixels((i, 0))
            end = self._grid_to_pixels((i, self.GRID_HEIGHT))
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for i in range(self.GRID_HEIGHT + 1):
            start = self._grid_to_pixels((0, i))
            end = self._grid_to_pixels((self.GRID_WIDTH, i))
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # Draw special shot
        if self.special_shot_effect:
            px, py = self._grid_to_pixels(self.special_shot_effect["center"])
            radius = int(self.special_shot_effect["radius"] * self.CELL_SIZE)
            color = (255, 100, 0) if self.special_shot_effect["hit"] else self.COLOR_SPECIAL_SHOT
            # Draw anti-aliased circle
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, color)
            if radius > 1:
                pygame.gfxdraw.aacircle(self.screen, px, py, radius-1, color)


        # Draw particles
        for p in self.particles:
            px, py = self._grid_to_pixels(p["pos"])
            alpha = max(0, int(255 * (p["life"] / 20)))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 2, 2))
            self.screen.blit(temp_surf, (px, py))

        # Draw paddle
        px, py = self._grid_to_pixels((self.paddle.x, self.paddle.y))
        pw, ph = self.paddle.width * self.CELL_SIZE, self.paddle.height * self.CELL_SIZE
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, (px, py, pw, ph))

        # Draw ball
        px, py = self._grid_to_pixels(self.ball_pos)
        pygame.draw.rect(self.screen, self.COLOR_BALL, (px, py, self.CELL_SIZE, self.CELL_SIZE))

        # Screen flash on miss
        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            alpha = int(128 * (self.screen_flash_timer / 10))
            flash_surface.fill((*self.COLOR_MISS_FLASH, alpha))
            self.screen.blit(flash_surface, (0, 0))
            self.screen_flash_timer -= 1

    def _render_ui(self):
        score_text = self.font_big.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.OFFSET_X, 10))
        
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - self.OFFSET_X - lives_text.get_width(), 10))

        special_text = self.font_small.render(f"SPECIAL: {self.special_shots}", True, self.COLOR_SPECIAL_SHOT)
        self.screen.blit(special_text, (self.SCREEN_WIDTH - self.OFFSET_X - special_text.get_width(), 35))

        if self.game_over:
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = (255,255,0) if self.score >= self.WIN_SCORE else (255,0,0)
            end_text = self.font_big.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

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


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # Use a persistent window for human play
    pygame.display.set_caption("Grid Pong")
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop for human play
    while not terminated:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
            
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        # In a real game, you might want to use keydown events for space
        # but for gym compatibility, we check the held state.
        
        action = np.array([movement, space_held, shift_held])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Control FPS
        
    print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
    env.close()