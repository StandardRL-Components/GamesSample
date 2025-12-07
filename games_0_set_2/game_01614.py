
# Generated: 2025-08-27T17:41:40.566829
# Source Brief: brief_01614.md
# Brief Index: 1614

        
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
        "Controls: ↑↓ to steer. Avoid the orange obstacles and stay on the blue track."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Race a car along a neon track, dodging obstacles to reach the finish line as quickly as possible."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.TRACK_LENGTH = 5000
        self.MAX_STEPS = 1000
        self.CHECKPOINT_INTERVAL = 500

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_TRACK = (0, 100, 255)
        self.COLOR_TRACK_GLOW = (0, 150, 255, 50)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLAYER_GLOW = (255, 50, 50, 100)
        self.COLOR_OBSTACLE = (255, 165, 0)
        self.COLOR_OBSTACLE_GLOW = (255, 165, 0, 80)
        self.COLOR_FINISH_LINE = (0, 255, 100)
        self.COLOR_TEXT = (240, 240, 255)

        # Initialize state variables
        self.car_pos = None
        self.car_velocity_y = None
        self.track_points = None
        self.obstacles = None
        self.particles = None
        self.camera_x = None
        self.obstacle_spawn_rate = None
        self.obstacle_speed_multiplier = None
        self.last_checkpoint_x = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.car_pos = pygame.Vector2(100, self.HEIGHT / 2)
        self.car_velocity_y = 0

        self._generate_track()
        self.obstacles = []
        self.particles = []

        self.camera_x = 0
        self.obstacle_spawn_rate = 0.05
        self.obstacle_speed_multiplier = 1.0

        self.last_checkpoint_x = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right

        reward = 0
        if not self.game_over:
            self._handle_input(movement)
            self._update_player()
            self._update_world()
            self._check_collisions()
            reward = self._calculate_reward()
            self._check_termination_conditions()

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True # Time ran out
            reward -= 50 # Penalty for running out of time
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,  # truncated always False
            self._get_info()
        )
    
    def _handle_input(self, movement):
        # sfx: player_accelerate
        if movement == 1:  # Up
            self.car_velocity_y -= 1.2
        elif movement == 2:  # Down
            self.car_velocity_y += 1.2

    def _update_player(self):
        # Apply damping
        self.car_velocity_y *= 0.9
        self.car_pos.y += self.car_velocity_y

        # Clamp position to screen bounds
        self.car_pos.y = max(10, min(self.HEIGHT - 10, self.car_pos.y))

    def _update_world(self):
        # Constant horizontal speed for the car by scrolling the world
        car_speed_x = 10
        self.camera_x += car_speed_x

        # Update and remove off-screen obstacles
        for obstacle in self.obstacles[:]:
            obstacle['rect'].x -= obstacle['speed'] * self.obstacle_speed_multiplier
            if obstacle['rect'].right < 0:
                self.obstacles.remove(obstacle)

        # Spawn new obstacles
        if self.np_random.random() < self.obstacle_spawn_rate:
            self._spawn_obstacle()

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
        
        # Increase difficulty
        if self.steps > 0 and self.steps % 200 == 0:
            self.obstacle_speed_multiplier += 0.05
        if self.steps > 0 and self.steps % 10 == 0:
            self.obstacle_spawn_rate = min(0.2, self.obstacle_spawn_rate + 0.001)

    def _spawn_obstacle(self):
        # Find track y-position at the spawn point
        spawn_x = self.camera_x + self.WIDTH + 50
        track_y = self._get_track_y_at(spawn_x)
        
        # Spawn obstacle near the track
        obstacle_y = track_y + self.np_random.uniform(-80, 80)
        obstacle_size = self.np_random.integers(15, 30)
        obstacle_rect = pygame.Rect(self.WIDTH + 50, obstacle_y - obstacle_size / 2, obstacle_size, obstacle_size)
        
        obstacle_speed = self.np_random.uniform(1, 3)

        self.obstacles.append({
            'rect': obstacle_rect,
            'speed': obstacle_speed,
            'pulse_phase': self.np_random.random() * math.pi * 2
        })

    def _check_collisions(self):
        car_rect = pygame.Rect(self.car_pos.x - 12, self.car_pos.y - 6, 24, 12)
        for obstacle in self.obstacles:
            if car_rect.colliderect(obstacle['rect']):
                self.game_over = True
                self.game_won = False
                # sfx: obstacle_crash
                self._create_particles(self.car_pos, self.COLOR_OBSTACLE, 50)
                self._create_particles(self.car_pos, self.COLOR_PLAYER, 30)
                break

    def _calculate_reward(self):
        reward = 0
        if self.game_over and not self.game_won:
            return -100.0 # Collision penalty
        
        # Survival reward
        reward += 0.1

        # Checkpoint reward
        if self.camera_x > self.last_checkpoint_x + self.CHECKPOINT_INTERVAL:
            self.last_checkpoint_x += self.CHECKPOINT_INTERVAL
            reward += 10.0
            # sfx: checkpoint_reached
            self._create_particles(pygame.Vector2(self.car_pos.x, self.car_pos.y), self.COLOR_FINISH_LINE, 20, is_checkpoint=True)

        return reward

    def _check_termination_conditions(self):
        if self.camera_x >= self.TRACK_LENGTH:
            self.game_over = True
            self.game_won = True
            self.score += 100 # Goal-oriented reward

    def _generate_track(self):
        points = []
        y = self.HEIGHT / 2
        segment_length = 50
        for x in range(0, self.TRACK_LENGTH + self.WIDTH, segment_length):
            points.append((x, y))
            y += self.np_random.uniform(-40, 40)
            y = max(80, min(self.HEIGHT - 80, y))
        self.track_points = points

    def _get_track_y_at(self, x_pos):
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i+1]
            if p1[0] <= x_pos < p2[0]:
                # Linear interpolation
                t = (x_pos - p1[0]) / (p2[0] - p1[0])
                return p1[1] + t * (p2[1] - p1[1])
        return self.HEIGHT / 2

    def _create_particles(self, pos, color, count, is_checkpoint=False):
        for _ in range(count):
            if is_checkpoint:
                angle = self.np_random.uniform(math.pi * 0.25, math.pi * 0.75) # Upwards burst
                speed = self.np_random.uniform(2, 5)
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 6)
            
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(20, 40),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_track()
        self._render_finish_line()
        self._render_obstacles()
        self._render_particles()
        if not (self.game_over and not self.game_won): # Don't render player if crashed
             self._render_player()

    def _render_track(self):
        screen_points = []
        for x, y in self.track_points:
            screen_x = x - self.camera_x
            if -50 < screen_x < self.WIDTH + 50:
                screen_points.append((screen_x, y))
        
        if len(screen_points) > 1:
            pygame.draw.lines(self.screen, self.COLOR_TRACK_GLOW, False, screen_points, 15)
            pygame.draw.lines(self.screen, self.COLOR_TRACK, False, screen_points, 5)

    def _render_finish_line(self):
        finish_x = self.TRACK_LENGTH - self.camera_x
        if -10 < finish_x < self.WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (finish_x, 0), (finish_x, self.HEIGHT), 5)
            # Add a glow
            glow_line = pygame.Surface((15, self.HEIGHT), pygame.SRCALPHA)
            glow_line.fill((*self.COLOR_FINISH_LINE, 30))
            self.screen.blit(glow_line, (finish_x - 7, 0))

    def _render_player(self):
        # Car body as a polygon
        p1 = (self.car_pos.x + 12, self.car_pos.y)
        p2 = (self.car_pos.x - 12, self.car_pos.y - 6)
        p3 = (self.car_pos.x - 12, self.car_pos.y + 6)
        points = [p1, p2, p3]

        # Glow effect
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER_GLOW)

        # Main body
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_obstacles(self):
        for obstacle in self.obstacles:
            rect = obstacle['rect']
            pulse = (math.sin(self.steps * 0.1 + obstacle['pulse_phase']) + 1) / 2
            
            # Glow
            glow_size = rect.width + 10 + pulse * 5
            glow_rect = pygame.Rect(0, 0, glow_size, glow_size)
            glow_rect.center = rect.center
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, self.COLOR_OBSTACLE_GLOW, glow_surface.get_rect(), border_radius=int(glow_size/4))
            self.screen.blit(glow_surface, glow_rect.topleft)

            # Main body
            pygame.gfxdraw.box(self.screen, rect, self.COLOR_OBSTACLE)

    def _render_particles(self):
        for p in self.particles:
            size = max(0, int(p['lifespan'] / 8))
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], p['pos'], size)

    def _render_ui(self):
        # Time display
        time_text = f"TIME: {self.steps / 30.0:.1f}s"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Score display
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 40))

        # Progress bar
        progress = self.camera_x / self.TRACK_LENGTH
        bar_width = 200
        bar_height = 10
        pygame.draw.rect(self.screen, (255,255,255,50), (self.WIDTH - bar_width - 10, 10, bar_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_FINISH_LINE, (self.WIDTH - bar_width - 10, 10, bar_width * progress, bar_height), border_radius=3)

        if self.game_over:
            if self.game_won:
                msg = "FINISH!"
                color = self.COLOR_FINISH_LINE
            else:
                msg = "CRASHED"
                color = self.COLOR_PLAYER
            
            over_surf = self.font_game_over.render(msg, True, color)
            over_rect = over_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(over_surf, over_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "camera_x": self.camera_x,
            "game_won": self.game_won
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Neon Line Racer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Action defaults
        movement = 0 # No-op
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # Get key presses for continuous actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Final Info: {info}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Match the auto-advance rate
        
    env.close()