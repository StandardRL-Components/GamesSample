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


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑/↓ to move the paddle. Press Space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Bounce a ball to break bricks and ascend a procedurally generated tower. "
        "Avoid hazards and try to reach the top!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.WORLD_HEIGHT = 8000
        self.INITIAL_LIVES = 3

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_PADDLE = (50, 200, 50)
        self.COLOR_PADDLE_OUTLINE = (150, 255, 150)
        self.COLOR_BALL = (255, 80, 80)
        self.COLOR_BALL_OUTLINE = (255, 200, 200)
        self.COLOR_BRICK = (80, 120, 255)
        self.COLOR_BRICK_OUTLINE = (180, 200, 255)
        self.COLOR_HAZARD = (255, 200, 0)
        self.COLOR_HAZARD_OUTLINE = (255, 255, 150)
        self.COLOR_ROAD_LINE = (50, 60, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)

        # Physics
        self.PADDLE_SPEED = 12
        self.BALL_BASE_SPEED = 8
        self.BALL_GRAVITY = 0.35
        self.MAX_BALL_SPEED_Y = 15

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
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # State variables (initialized to avoid errors)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle = None
        self.ball = None
        self.bricks = []
        self.hazards = []
        self.particles = []
        self.camera_y = 0.0
        self.max_progress = 0.0
        self.last_generated_y = 0
        self.hazard_chance = 0.1 # Initial hazard chance
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - 50, self.HEIGHT - 40, 100, 15
        )
        
        self.ball = {
            "world_pos": pygame.Vector2(self.paddle.centerx, self.paddle.top - 10),
            "vel": pygame.Vector2(0, 0),
            "radius": 8,
            "launched": False,
        }
        
        self.bricks = []
        self.hazards = []
        self.particles = []
        self.camera_y = 0.0
        self.max_progress = 0.0
        self.last_generated_y = 0
        self.hazard_chance = 0.1
        
        self._generate_level_chunk(0, self.HEIGHT * 2)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Time penalty

        # 1. Handle Input
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held)

        # 2. Update Game Logic
        events = self._update_ball()
        
        # 3. Calculate rewards based on events
        if "brick_broken" in events:
            reward += 1.0
        if "hazard_hit" in events:
            self.score = max(0, self.score - 50) # Score penalty
            reward -= 5.0
        if "life_lost" in events:
            self.lives -= 1
            reward -= 10.0
            if self.lives <= 0:
                self.game_over = True

        # 4. Update Camera and World
        self._update_camera()
        self._generate_new_chunks()
        self._update_particles()
        
        # 5. Update difficulty
        if self.steps > 0 and self.steps % 200 == 0:
            self.hazard_chance = min(0.5, self.hazard_chance * 1.01)

        # 6. Check Termination
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.ball["world_pos"].y >= self.WORLD_HEIGHT:
            reward += 100.0
            terminated = True
            self.game_over = True # To show victory message

        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        if movement == 1: # Up
            self.paddle.y -= self.PADDLE_SPEED
        elif movement == 2: # Down
            self.paddle.y += self.PADDLE_SPEED
        
        self.paddle.top = max(self.HEIGHT/2, self.paddle.top)
        self.paddle.bottom = min(self.HEIGHT, self.paddle.bottom)

        if space_held and not self.ball["launched"]:
            self.ball["launched"] = True
            angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            self.ball["vel"] = pygame.Vector2(
                math.cos(angle) * self.BALL_BASE_SPEED,
                math.sin(angle) * self.BALL_BASE_SPEED
            )

    def _update_ball(self):
        if not self.ball["launched"]:
            self.ball["world_pos"].x = self.paddle.centerx
            self.ball["world_pos"].y = self.paddle.top - self.ball["radius"]
            return {}

        events = {}
        
        # Apply gravity
        self.ball["vel"].y += self.BALL_GRAVITY
        self.ball["vel"].y = min(self.ball["vel"].y, self.MAX_BALL_SPEED_Y)
        
        self.ball["world_pos"] += self.ball["vel"]
        ball_rect = pygame.Rect(
            self.ball["world_pos"].x - self.ball["radius"],
            self.ball["world_pos"].y - self.ball["radius"],
            self.ball["radius"] * 2,
            self.ball["radius"] * 2
        )

        # Wall collisions
        if ball_rect.left <= 0:
            ball_rect.left = 0
            self.ball["vel"].x *= -1
        if ball_rect.right >= self.WIDTH:
            ball_rect.right = self.WIDTH
            self.ball["vel"].x *= -1

        # Paddle collision
        paddle_screen_rect = self.paddle.copy()
        ball_screen_pos_y = self.ball["world_pos"].y - self.camera_y
        ball_screen_rect = pygame.Rect(ball_rect.x, ball_screen_pos_y - self.ball['radius'], ball_rect.width, ball_rect.height)

        if ball_screen_rect.colliderect(paddle_screen_rect) and self.ball["vel"].y > 0:
            self.ball["vel"].y *= -1.05 # Add some energy
            
            offset = (self.ball["world_pos"].x - self.paddle.centerx) / (self.paddle.width / 2)
            self.ball["vel"].x += offset * 3
            
            self.ball["vel"].x = max(-self.BALL_BASE_SPEED, min(self.BALL_BASE_SPEED, self.ball["vel"].x))
            self.ball["vel"].y = max(-self.MAX_BALL_SPEED_Y, min(self.MAX_BALL_SPEED_Y, self.ball["vel"].y))
            
            ball_screen_rect.bottom = paddle_screen_rect.top
            self.ball["world_pos"].y = ball_screen_rect.centery + self.camera_y
            
            self.score += 1
            
        # Brick collisions
        for brick in self.bricks[:]:
            if ball_rect.colliderect(brick):
                self.bricks.remove(brick)
                self.score += 10
                events["brick_broken"] = True
                
                for _ in range(15):
                    self._create_particle(brick.center, self.COLOR_BRICK)

                self.ball["vel"].y *= -1
                break

        # Hazard collisions
        for hazard in self.hazards[:]:
            if ball_rect.colliderect(hazard):
                self.hazards.remove(hazard)
                events["hazard_hit"] = True
                self.ball["vel"].y *= -1.1
                for _ in range(20):
                    self._create_particle(hazard.center, self.COLOR_HAZARD)

        # Fall off screen
        if ball_screen_pos_y > self.HEIGHT + self.ball["radius"]:
            events["life_lost"] = True
            self.ball["launched"] = False
            self.ball["vel"] = pygame.Vector2(0, 0)

        self.ball["world_pos"].x = ball_rect.centerx
        self.ball["world_pos"].y = ball_rect.centery
        return events

    def _update_camera(self):
        target_camera_y = self.ball["world_pos"].y - self.HEIGHT * 0.4
        self.camera_y = max(self.camera_y, target_camera_y)
        self.max_progress = max(self.max_progress, self.ball["world_pos"].y)
    
    def _generate_new_chunks(self):
        if self.camera_y + self.HEIGHT > self.last_generated_y:
            self._generate_level_chunk(self.last_generated_y, self.last_generated_y + self.HEIGHT)

    def _generate_level_chunk(self, y_start, y_end):
        for y in range(int(y_start), int(y_end), 80):
            if y > self.WORLD_HEIGHT - 200:
                break
            if y < 200:
                continue
                
            row_type = self.np_random.choice(['bricks', 'hazard', 'empty'])
            
            if row_type == 'bricks':
                num_bricks = self.np_random.integers(3, 7)
                brick_positions = self.np_random.choice(range(8), size=num_bricks, replace=False)
                for i in brick_positions:
                    brick_x = i * (self.WIDTH / 8) + 10
                    self.bricks.append(pygame.Rect(brick_x, y, self.WIDTH/8 - 20, 20))

            elif row_type == 'hazard' and self.np_random.random() < self.hazard_chance:
                num_hazards = self.np_random.integers(1, 3)
                hazard_positions = self.np_random.choice(range(10), size=num_hazards, replace=False)
                for i in hazard_positions:
                    hazard_x = i * (self.WIDTH / 10) + 5
                    self.hazards.append(pygame.Rect(hazard_x, y, self.WIDTH/10 - 10, 15))

        self.last_generated_y = y_end

    def _create_particle(self, pos, color):
        particle = {
            "pos": pygame.Vector2(pos),
            "vel": pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-4, 1)),
            "lifespan": self.np_random.integers(15, 30),
            "color": color,
            "radius": self.np_random.uniform(2, 5)
        }
        self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"].y += 0.2
            p["lifespan"] -= 1
            p["radius"] -= 0.1
            if p["lifespan"] <= 0 or p["radius"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        line_y_start = - (self.camera_y % 50)
        for i in range(self.HEIGHT // 50 + 2):
            y = line_y_start + i * 50
            pygame.draw.line(self.screen, self.COLOR_ROAD_LINE, (self.WIDTH / 2, y), (self.WIDTH / 2, y + 25), 3)

        for brick in self.bricks:
            screen_rect = brick.move(0, -self.camera_y)
            if screen_rect.bottom > 0 and screen_rect.top < self.HEIGHT:
                pygame.draw.rect(self.screen, self.COLOR_BRICK, screen_rect, border_radius=3)
                pygame.draw.rect(self.screen, self.COLOR_BRICK_OUTLINE, screen_rect, 2, border_radius=3)

        for hazard in self.hazards:
            screen_rect = hazard.move(0, -self.camera_y)
            if screen_rect.bottom > 0 and screen_rect.top < self.HEIGHT:
                pygame.draw.rect(self.screen, self.COLOR_HAZARD, screen_rect, border_radius=5)
                pygame.draw.rect(self.screen, self.COLOR_HAZARD_OUTLINE, screen_rect, 2, border_radius=5)
        
        for p in self.particles:
            screen_pos = p["pos"] - pygame.Vector2(0, self.camera_y)
            alpha = max(0, min(255, int(255 * (p["lifespan"] / 20))))
            color = (*p["color"], alpha)
            radius = int(p["radius"])
            if radius <= 0: continue
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, color)
            self.screen.blit(temp_surf, (int(screen_pos.x - radius), int(screen_pos.y - radius)))

        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_OUTLINE, self.paddle, 2, border_radius=3)
        
        ball_screen_pos = self.ball["world_pos"] - pygame.Vector2(0, self.camera_y)
        pygame.gfxdraw.aacircle(self.screen, int(ball_screen_pos.x), int(ball_screen_pos.y), int(self.ball["radius"]), self.COLOR_BALL_OUTLINE)
        pygame.gfxdraw.filled_circle(self.screen, int(ball_screen_pos.x), int(ball_screen_pos.y), int(self.ball["radius"]), self.COLOR_BALL)
        
        finish_y = self.WORLD_HEIGHT - self.camera_y
        if 0 < finish_y < self.HEIGHT:
            pygame.draw.line(self.screen, (255, 255, 255), (0, finish_y), (self.WIDTH, finish_y), 5)


    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (12, 12))
        
        for i in range(self.lives):
            pos = (self.WIDTH - 30 - i * 25, 15)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_BALL_OUTLINE)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_BALL)

        progress_percent = self.max_progress / self.WORLD_HEIGHT
        bar_width = self.WIDTH - 20
        bar_height = 5
        pygame.draw.rect(self.screen, self.COLOR_ROAD_LINE, (10, self.HEIGHT - 15, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, (10, self.HEIGHT - 15, bar_width * progress_percent, bar_height))

        if self.game_over:
            msg = "VICTORY!" if self.lives > 0 else "GAME OVER"
            color = (100, 255, 100) if self.lives > 0 else (255, 100, 100)
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "progress": self.max_progress / self.WORLD_HEIGHT
        }
    
    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # To play the game manually, you need to have pygame installed and a display.
    # The environment itself runs headlessly, but this block is for human interaction.
    try:
        # Unset the dummy video driver for manual play
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
        
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        pygame.display.set_caption("Brick Breaker Ascent")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        running = True
        
        total_reward = 0
        
        while running:
            movement = 0 # no-op
            space_held = 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                movement = 2
            
            if keys[pygame.K_SPACE]:
                space_held = 1
            
            action = [movement, space_held, 0] # Last action element is unused
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0
                
        env.close()
    except pygame.error as e:
        print("Could not run manual test, likely because no display is available.")
        print(f"Pygame error: {e}")