
# Generated: 2025-08-28T06:25:21.218677
# Source Brief: brief_02925.md
# Brief Index: 2925

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: ↑↓ to steer. Hold Space to accelerate, hold Shift to brake."
    )

    game_description = (
        "Race against time in a procedurally generated neon track, dodging obstacles to reach the finish line."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 10, 30)
    COLOR_TRACK = (0, 150, 255)
    COLOR_TRACK_GLOW = (0, 150, 255, 50)
    COLOR_CAR = (0, 255, 100)
    COLOR_CAR_GLOW = (0, 255, 100, 100)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (255, 50, 50, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_FINISH_1 = (255, 255, 255)
    COLOR_FINISH_2 = (0, 0, 0)
    
    # Physics & Gameplay
    CAR_INITIAL_X = 100
    CAR_STEER_SPEED = 4.0
    CAR_ACCELERATION = 0.05
    CAR_DECELERATION = 0.1
    CAR_MIN_SPEED = 1.0
    CAR_MAX_SPEED = 8.0
    CAR_INITIAL_SPEED = 2.0
    
    TRACK_Y_TOP = 80
    TRACK_Y_BOTTOM = 320
    TRACK_FINISH_DISTANCE = 15000
    
    OBSTACLE_SIZE = 20
    MAX_COLLISIONS = 5
    MAX_STEPS = 7200  # 120 seconds at 60fps
    
    # Difficulty Scaling
    INITIAL_OBSTACLE_SPEED = 2.0
    INITIAL_OBSTACLE_SPAWN_PROB = 0.01
    
    # Rewards
    REWARD_SURVIVE = 0.01
    REWARD_PASS_OBSTACLE = 1.0
    REWARD_COLLISION = -5.0
    REWARD_FINISH = 100.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Consolas', 24, bold=True)
        
        # Uninitialized state variables
        self.car_pos = None
        self.car_speed = None
        self.distance_traveled = None
        self.obstacles = None
        self.collision_count = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.obstacle_speed = None
        self.obstacle_spawn_prob = None
        self.particles = None
        self.collision_flash = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.car_pos = pygame.Vector2(self.CAR_INITIAL_X, self.SCREEN_HEIGHT / 2)
        self.car_speed = self.CAR_INITIAL_SPEED
        self.distance_traveled = 0
        self.obstacles = []
        self.collision_count = 0
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.obstacle_spawn_prob = self.INITIAL_OBSTACLE_SPAWN_PROB
        
        self.particles = []
        self.collision_flash = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = self.REWARD_SURVIVE
        
        self._handle_input(action)
        self._update_game_state()
        
        collision_reward, passed_reward = self._handle_obstacles_and_collisions()
        reward += collision_reward
        reward += passed_reward
        
        self.steps += 1
        self.score += reward
        
        terminated = self._check_termination()
        if terminated and self.distance_traveled >= self.TRACK_FINISH_DISTANCE:
            reward += self.REWARD_FINISH
            self.score += self.REWARD_FINISH

        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if movement == 1:  # Up
            self.car_pos.y -= self.CAR_STEER_SPEED
        if movement == 2:  # Down
            self.car_pos.y += self.CAR_STEER_SPEED
            
        if space_held:
            self.car_speed = min(self.CAR_MAX_SPEED, self.car_speed + self.CAR_ACCELERATION)
        if shift_held:
            self.car_speed = max(self.CAR_MIN_SPEED, self.car_speed - self.CAR_DECELERATION)

        self.car_pos.y = np.clip(self.car_pos.y, self.TRACK_Y_TOP + 15, self.TRACK_Y_BOTTOM - 15)

    def _update_game_state(self):
        self.distance_traveled += self.car_speed
        
        # Difficulty scaling
        if self.steps % 200 == 0 and self.steps > 0:
            self.obstacle_speed += 0.2
        if self.steps % 100 == 0 and self.steps > 0:
            self.obstacle_spawn_prob = min(0.05, self.obstacle_spawn_prob + 0.001)

        # Spawn new obstacles
        if self.np_random.random() < self.obstacle_spawn_prob:
            obstacle_y = self.np_random.integers(self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM - self.OBSTACLE_SIZE)
            new_obstacle = {
                "rect": pygame.Rect(self.SCREEN_WIDTH, obstacle_y, self.OBSTACLE_SIZE, self.OBSTACLE_SIZE),
                "passed": False
            }
            self.obstacles.append(new_obstacle)
            
        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        if self.collision_flash > 0:
            self.collision_flash -= 1

    def _handle_obstacles_and_collisions(self):
        collision_reward = 0
        passed_reward = 0
        car_rect = pygame.Rect(self.car_pos.x - 10, self.car_pos.y - 7, 20, 14)

        for obs in self.obstacles[:]:
            obs['rect'].x -= self.obstacle_speed + self.car_speed
            
            if obs['rect'].colliderect(car_rect):
                # sound: explosion
                self.collision_count += 1
                collision_reward += self.REWARD_COLLISION
                self.obstacles.remove(obs)
                self.car_speed = 0 # Penalty
                self.collision_flash = 10
                self._create_particles(obs['rect'].center)
                continue

            if not obs['passed'] and obs['rect'].right < self.car_pos.x:
                obs['passed'] = True
                passed_reward += self.REWARD_PASS_OBSTACLE
                
            if obs['rect'].right < 0:
                self.obstacles.remove(obs)
        
        return collision_reward, passed_reward
        
    def _create_particles(self, pos):
        for _ in range(20):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': self.COLOR_OBSTACLE
            })

    def _check_termination(self):
        return (self.collision_count >= self.MAX_COLLISIONS or 
                self.steps >= self.MAX_STEPS or
                self.distance_traveled >= self.TRACK_FINISH_DISTANCE)

    def _get_observation(self):
        bg_color = self.COLOR_BG
        if self.collision_flash > 0:
            t = self.collision_flash / 10
            bg_color = (
                int(self.COLOR_BG[0] + (80 - self.COLOR_BG[0]) * t),
                self.COLOR_BG[1],
                self.COLOR_BG[2]
            )
        self.screen.fill(bg_color)
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_track_and_finish()
        self._render_obstacles()
        self._render_car()
        self._render_particles()

    def _render_track_and_finish(self):
        # Track lines
        for y_pos in [self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM]:
            pygame.draw.line(self.screen, self.COLOR_TRACK, (0, y_pos), (self.SCREEN_WIDTH, y_pos), 3)
            pygame.gfxdraw.line(self.screen, 0, y_pos, self.SCREEN_WIDTH, y_pos, self.COLOR_TRACK_GLOW)
        
        # Finish line
        if self.distance_traveled > self.TRACK_FINISH_DISTANCE - self.SCREEN_WIDTH * 2:
            finish_x = self.SCREEN_WIDTH + (self.TRACK_FINISH_DISTANCE - self.distance_traveled)
            if finish_x < self.SCREEN_WIDTH:
                check_size = 20
                for y in range(self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM, check_size):
                    for x_offset in range(0, 2 * check_size, check_size):
                        color = self.COLOR_FINISH_1 if ((y // check_size) % 2 == (x_offset // check_size) % 2) else self.COLOR_FINISH_2
                        rect = pygame.Rect(finish_x + x_offset, y, check_size, check_size)
                        pygame.draw.rect(self.screen, color, rect)

    def _render_obstacles(self):
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'])
            glow_rect = obs['rect'].inflate(4, 4)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_GLOW, glow_rect, 2)

    def _render_car(self):
        p1 = self.car_pos + pygame.Vector2(15, 0)
        p2 = self.car_pos + pygame.Vector2(-15, -10)
        p3 = self.car_pos + pygame.Vector2(-15, 10)
        points = [p1, p2, p3]
        
        # Glow effect
        pygame.gfxdraw.aapolygon(self.screen, [(int(p.x), int(p.y)) for p in points], self.COLOR_CAR_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in points], self.COLOR_CAR_GLOW)

        # Main car shape
        pygame.gfxdraw.aapolygon(self.screen, [(int(p.x), int(p.y)) for p in points], self.COLOR_CAR)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in points], self.COLOR_CAR)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            size = int(p['life'] / 5)
            pygame.draw.circle(self.screen, color, (int(p['pos'].x), int(p['pos'].y)), size)

    def _render_ui(self):
        # Collisions
        hits_text = self.font.render(f"HITS: {self.collision_count}/{self.MAX_COLLISIONS}", True, self.COLOR_TEXT)
        self.screen.blit(hits_text, (10, 10))
        
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 60
        timer_text = self.font.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Progress bar
        progress_ratio = self.distance_traveled / self.TRACK_FINISH_DISTANCE
        bar_width = self.SCREEN_WIDTH - 20
        bar_height = 10
        
        pygame.draw.rect(self.screen, (50, 50, 80), (10, self.SCREEN_HEIGHT - 20, bar_width, bar_height))
        fill_width = min(bar_width, int(bar_width * progress_ratio))
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (10, self.SCREEN_HEIGHT - 20, fill_width, bar_height))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_traveled": self.distance_traveled,
            "collision_count": self.collision_count
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
    
    pygame.display.set_caption("Neon Racer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        if keys[pygame.K_ESCAPE]:
            running = False
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Distance: {info['distance_traveled']:.0f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting
            
        clock.tick(60)
        
    env.close()