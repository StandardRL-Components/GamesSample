import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to steer the car and avoid obstacles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced retro racer. Dodge obstacles and race against the clock to reach the finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_TRACK = (50, 50, 60)
    COLOR_TRACK_LINES = (80, 80, 90)
    COLOR_CAR = (255, 80, 80)
    COLOR_CAR_HIGHLIGHT = (255, 150, 150)
    COLOR_OBSTACLE = (80, 150, 255)
    COLOR_OBSTACLE_OUTLINE = (150, 200, 255)
    COLOR_PARTICLE = (255, 180, 50)
    COLOR_TEXT = (240, 240, 240)
    COLOR_FINISH_LINE_1 = (255, 255, 255)
    COLOR_FINISH_LINE_2 = (0, 0, 0)
    
    # Game Parameters
    GAME_DURATION_SECONDS = 30
    MAX_STEPS = GAME_DURATION_SECONDS * FPS + 30 # A little buffer
    DISTANCE_TO_FINISH = 6000
    
    # Car Physics
    CAR_WIDTH, CAR_HEIGHT = 25, 40
    CAR_Y_POS = SCREEN_HEIGHT - 80
    CAR_ACCELERATION = 0.8
    CAR_FRICTION = 0.90
    CAR_MAX_SPEED = 8.0
    
    # World & Difficulty
    INITIAL_SCROLL_SPEED = 6.0
    SCROLL_SPEED_INCREASE_RATE = 0.15 # per 5 seconds
    INITIAL_SPAWN_RATE = 2.0 # obstacles per second
    SPAWN_RATE_INCREASE = 0.1 # per second
    
    # Obstacles
    OBSTACLE_MIN_SIZE = 20
    OBSTACLE_MAX_SIZE = 45
    
    # Rewards
    REWARD_WIN = 100.0
    REWARD_CRASH = -100.0
    REWARD_TIMEOUT = -10.0
    REWARD_SURVIVAL = 0.01 # Small reward for each step survived
    REWARD_CHECKPOINT = 1.0
    CHECKPOINT_DISTANCE = 200 # pixels

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 50, bold=True)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.terminated = False
        self.truncated = False
        self.game_outcome = ""

        self.car_pos_x = self.SCREEN_WIDTH / 2
        self.car_vel_x = 0.0
        
        self.obstacles = []
        self.particles = []
        
        self.distance_traveled = 0.0
        self.last_checkpoint_dist = 0.0
        
        self.game_timer = self.GAME_DURATION_SECONDS * self.FPS
        self.scroll_speed = self.INITIAL_SCROLL_SPEED
        self.obstacle_spawn_rate = self.INITIAL_SPAWN_RATE
        self.obstacle_spawn_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.terminated or self.truncated:
            return self._get_observation(), 0, self.terminated, self.truncated, self._get_info()

        reward = 0

        # 1. Handle Input & Update Car
        self._handle_input(action)
        self._update_car()

        # 2. Update World State
        self._update_world()
        self._update_particles()
        
        # 3. Calculate Rewards
        reward += self.REWARD_SURVIVAL
        if self.distance_traveled // self.CHECKPOINT_DISTANCE > self.last_checkpoint_dist // self.CHECKPOINT_DISTANCE:
            self.last_checkpoint_dist = self.distance_traveled
            reward += self.REWARD_CHECKPOINT

        # 4. Check Termination Conditions
        if self._check_collision():
            self.terminated = True
            reward = self.REWARD_CRASH
            self.game_outcome = "CRASH!"
            self._create_explosion(pygame.Vector2(self.car_pos_x, self.CAR_Y_POS))
        elif self.distance_traveled >= self.DISTANCE_TO_FINISH:
            self.terminated = True
            reward = self.REWARD_WIN
            self.game_outcome = "FINISH!"
        elif self.game_timer <= 0 or self.steps >= self.MAX_STEPS:
            self.truncated = True
            reward = self.REWARD_TIMEOUT
            self.game_outcome = "TIME UP"

        self.score += reward
        self.steps += 1
        
        return self._get_observation(), reward, self.terminated, self.truncated, self._get_info()

    def _handle_input(self, action):
        movement = action[0]
        
        if movement == 3: # Left
            self.car_vel_x -= self.CAR_ACCELERATION
        elif movement == 4: # Right
            self.car_vel_x += self.CAR_ACCELERATION

    def _update_car(self):
        # Apply friction
        self.car_vel_x *= self.CAR_FRICTION
        self.car_vel_x = np.clip(self.car_vel_x, -self.CAR_MAX_SPEED, self.CAR_MAX_SPEED)
        
        # Update position
        self.car_pos_x += self.car_vel_x
        
        # Keep car within track bounds (approx 100px margin)
        track_width = self.SCREEN_WIDTH - 200
        self.car_pos_x = np.clip(self.car_pos_x, 100 + self.CAR_WIDTH/2, self.SCREEN_WIDTH - 100 - self.CAR_WIDTH/2)

    def _update_world(self):
        # Update timers and difficulty
        self.game_timer -= 1
        if self.steps > 0 and self.steps % (5 * self.FPS) == 0:
            self.scroll_speed += self.SCROLL_SPEED_INCREASE_RATE
        if self.steps > 0 and self.steps % self.FPS == 0:
            self.obstacle_spawn_rate += self.SPAWN_RATE_INCREASE

        # Update world scroll
        self.distance_traveled += self.scroll_speed

        # Update obstacles
        for obstacle in self.obstacles:
            obstacle['rect'].y += self.scroll_speed
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].top < self.SCREEN_HEIGHT]

        # Spawn new obstacles
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            self.obstacle_spawn_timer = self.FPS / self.obstacle_spawn_rate
            self._spawn_obstacle()

    def _spawn_obstacle(self):
        size = self.np_random.integers(self.OBSTACLE_MIN_SIZE, self.OBSTACLE_MAX_SIZE)
        track_width = self.SCREEN_WIDTH - 240 # Playable area
        x = 120 + self.np_random.random() * (track_width - size)
        y = -size
        
        rect = pygame.Rect(x, y, size, size)
        shape = 'rect' if self.np_random.random() > 0.5 else 'circle'
        
        self.obstacles.append({'rect': rect, 'shape': shape})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _check_collision(self):
        car_rect = pygame.Rect(self.car_pos_x - self.CAR_WIDTH/2, self.CAR_Y_POS - self.CAR_HEIGHT/2, self.CAR_WIDTH, self.CAR_HEIGHT)
        for obs in self.obstacles:
            if car_rect.colliderect(obs['rect']):
                return True
        return False

    def _create_explosion(self, pos):
        for _ in range(30):
            angle = self.np_random.random() * 2 * math.pi
            speed = 2 + self.np_random.random() * 4
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(15, 30),
                'size': self.np_random.integers(2, 6)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Track
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (100, 0, self.SCREEN_WIDTH - 200, self.SCREEN_HEIGHT))
        
        # Scrolling track lines
        scroll_offset = self.distance_traveled % 80
        for i in range(-1, self.SCREEN_HEIGHT // 80 + 2):
            y = i * 80 + scroll_offset
            pygame.draw.line(self.screen, self.COLOR_TRACK_LINES, (100, y), (self.SCREEN_WIDTH-100, y), 3)

        # Finish line
        finish_line_screen_y = (self.DISTANCE_TO_FINISH - self.distance_traveled) + self.CAR_Y_POS
        if 0 < finish_line_screen_y < self.SCREEN_HEIGHT + 50:
             for i in range(100, self.SCREEN_WIDTH - 100, 20):
                color = self.COLOR_FINISH_LINE_1 if (i // 20) % 2 == 0 else self.COLOR_FINISH_LINE_2
                pygame.draw.rect(self.screen, color, (i, finish_line_screen_y, 20, 10))

        # Obstacles
        for obs in self.obstacles:
            r = obs['rect']
            if obs['shape'] == 'circle':
                pygame.gfxdraw.aacircle(self.screen, int(r.centerx), int(r.centery), int(r.width/2), self.COLOR_OBSTACLE_OUTLINE)
                pygame.gfxdraw.filled_circle(self.screen, int(r.centerx), int(r.centery), int(r.width/2), self.COLOR_OBSTACLE)
            else:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, r)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, r, 2)
        
        # Particles
        for p in self.particles:
            size = max(0, p['size'] * (p['lifespan'] / 30.0))
            pygame.draw.rect(self.screen, self.COLOR_PARTICLE, (p['pos'].x - size/2, p['pos'].y - size/2, size, size))

        # Car
        if not (self.terminated and self.game_outcome == "CRASH!"):
            car_rect = pygame.Rect(self.car_pos_x - self.CAR_WIDTH/2, self.CAR_Y_POS - self.CAR_HEIGHT/2, self.CAR_WIDTH, self.CAR_HEIGHT)
            highlight_rect = pygame.Rect(car_rect.left + 3, car_rect.top + 3, self.CAR_WIDTH - 6, self.CAR_HEIGHT - 6)
            pygame.draw.rect(self.screen, self.COLOR_CAR, car_rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_CAR_HIGHLIGHT, highlight_rect, border_radius=4)


    def _render_ui(self):
        # Timer
        time_left = max(0, self.game_timer / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (10, 10))
        
        # Speed
        speed_kmh = int(self.scroll_speed * 15) # Arbitrary multiplier for "speed" feel
        speed_text = self.font_ui.render(f"SPEED: {speed_kmh} KM/H", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (self.SCREEN_WIDTH - speed_text.get_width() - 10, 10))
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 35))

        # Game Over Message
        if self.terminated or self.truncated:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_big.render(self.game_outcome, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_traveled": self.distance_traveled,
            "time_left": max(0, self.game_timer / self.FPS)
        }

    def close(self):
        pygame.quit()
        
if __name__ == "__main__":
    env = GameEnv()
    
    # --- To play manually ---
    # This requires setting up a pygame display window
    
    # 1. Re-init with a display
    pygame.quit()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    env.screen = screen # Override the surface with the display
    obs, info = env.reset()
    
    # Game loop
    running = True
    while running:
        # Action defaults to no-op
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        if keys[pygame.K_r]: # Reset on 'R' key
            obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Outcome: {info.get('game_outcome', 'N/A')}. Resetting.")
            obs, info = env.reset()

        # The step function already renders to the env.screen surface,
        # so we just need to flip the display.
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(env.FPS)

    env.close()