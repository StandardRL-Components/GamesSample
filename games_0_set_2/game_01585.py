
# Generated: 2025-08-28T02:04:11.548800
# Source Brief: brief_01585.md
# Brief Index: 1585

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ for a small jump, ↓ for a large jump. The robot runs automatically."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a neon robot through a procedural obstacle course. Time your jumps to perfection to reach the finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FINISH_DISTANCE = 10000  # Total distance to travel in pixels
        self.MAX_STEPS = 3000
        self.ROBOT_SPEED = 4.0
        self.GRAVITY = 0.5
        self.SMALL_JUMP_POWER = -9.5
        self.LARGE_JUMP_POWER = -13.0
        self.GROUND_Y = self.HEIGHT - 60
        self.ROBOT_X_POS = 120

        # --- Colors ---
        self.COLOR_BG = (10, 10, 25)
        self.COLOR_GRID = (25, 25, 60)
        self.COLOR_GROUND = (40, 40, 70)
        self.COLOR_ROBOT_MAIN = (0, 200, 255)
        self.COLOR_ROBOT_ACCENT = (200, 255, 255)
        self.OBSTACLE_COLORS = [(255, 0, 100), (255, 100, 0), (255, 200, 0)] # Red, Orange, Yellow
        self.COLOR_FINISH = (0, 255, 120)
        self.COLOR_TEXT = (230, 230, 255)
        self.COLOR_TEXT_SHADOW = (20, 20, 40)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont("Consolas, 'Courier New', monospace", 30)
            self.font_small = pygame.font.SysFont("Consolas, 'Courier New', monospace", 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 40)
            self.font_small = pygame.font.Font(None, 24)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.distance_traveled = 0.0
        self.stage = 1
        self.robot_pos = [0, 0]
        self.robot_vel_y = 0.0
        self.on_ground = True
        self.obstacles = []
        self.particles = []
        self.cleared_obstacles = set()
        self.next_obstacle_spawn_dist = 0.0
        self.obstacle_id_counter = 0
        self.grid_offset = 0.0

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.distance_traveled = 0.0
        self.stage = 1
        self.robot_pos = [self.ROBOT_X_POS, self.GROUND_Y]
        self.robot_vel_y = 0.0
        self.on_ground = True
        self.obstacles = []
        self.particles = []
        self.cleared_obstacles = set()
        self.obstacle_id_counter = 0
        self.grid_offset = 0.0
        
        # Initial obstacle spawn distance
        self.next_obstacle_spawn_dist = self.np_random.uniform(400, 500)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, we still advance the clock for animations
            self.clock.tick(30)
            self._update_particles()
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.clock.tick(30)
        
        self._handle_input(action)
        self._update_player_state()
        self._update_world_state()
        self._update_particles()

        reward, terminated = self._process_events()

        self.score += reward
        self.steps += 1
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement = action[0]
        if self.on_ground:
            if movement == 1:  # Up Arrow -> Small Jump
                self.robot_vel_y = self.SMALL_JUMP_POWER
                self.on_ground = False
                # // Jump sound
                self._spawn_particles(self.robot_pos, 'jump', 15)
            elif movement == 2:  # Down Arrow -> Large Jump
                self.robot_vel_y = self.LARGE_JUMP_POWER
                self.on_ground = False
                # // Big jump sound
                self._spawn_particles(self.robot_pos, 'jump', 25)

    def _update_player_state(self):
        if not self.on_ground:
            self.robot_vel_y += self.GRAVITY
        
        self.robot_pos[1] += self.robot_vel_y

        if self.robot_pos[1] >= self.GROUND_Y:
            if not self.on_ground: # Just landed
                # // Landing sound
                self._spawn_particles([self.robot_pos[0], self.GROUND_Y], 'land', 20)
            self.robot_pos[1] = self.GROUND_Y
            self.robot_vel_y = 0
            self.on_ground = True
        
        self.robot_pos[1] = max(0, self.robot_pos[1])

    def _update_world_state(self):
        self.distance_traveled += self.ROBOT_SPEED
        self.grid_offset = (self.grid_offset - self.ROBOT_SPEED * 0.5) % 40

        # Update stage based on progress
        if self.distance_traveled > self.FINISH_DISTANCE * 0.66:
            self.stage = 3
        elif self.distance_traveled > self.FINISH_DISTANCE * 0.33:
            self.stage = 2

        # Move obstacles
        for obs in self.obstacles:
            obs['x'] -= self.ROBOT_SPEED
        
        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['x'] + obs['w'] > 0]

        # Spawn new obstacles
        self.next_obstacle_spawn_dist -= self.ROBOT_SPEED
        if self.next_obstacle_spawn_dist <= 0:
            self._spawn_obstacle()
            stage_multiplier = 1.0 - (self.stage - 1) * 0.2
            min_dist = 280 * stage_multiplier
            max_dist = 450 * stage_multiplier
            self.next_obstacle_spawn_dist = self.np_random.uniform(min_dist, max_dist)

    def _spawn_obstacle(self):
        obstacle_type_roll = self.np_random.random()
        
        if self.stage == 1: # Low only
            height_type = 0
        elif self.stage == 2: # Low and Medium
            height_type = 0 if obstacle_type_roll < 0.6 else 1
        else: # All three
            if obstacle_type_roll < 0.4: height_type = 0
            elif obstacle_type_roll < 0.8: height_type = 1
            else: height_type = 2
        
        heights = [25, 55, 85] # low, medium, high
        h = heights[height_type]
        w = self.np_random.integers(30, 60)
        
        self.obstacles.append({
            'id': self.obstacle_id_counter,
            'x': self.WIDTH,
            'y': self.GROUND_Y - h,
            'w': w,
            'h': h,
            'color': self.OBSTACLE_COLORS[height_type]
        })
        self.obstacle_id_counter += 1

    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # particle gravity
            p['life'] -= 1
            if p['life'] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def _process_events(self):
        reward = 0.1  # Survival reward per step
        terminated = False

        # Check for finish line
        if self.distance_traveled >= self.FINISH_DISTANCE:
            bonus = 100.0 * max(0, self.MAX_STEPS - self.steps) / self.MAX_STEPS
            reward += bonus
            terminated = True
            return reward, terminated

        robot_rect = pygame.Rect(self.robot_pos[0] - 12, self.robot_pos[1] - 35, 24, 35)

        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['x'], obs['y'], obs['w'], obs['h'])
            
            # Check collision
            if robot_rect.colliderect(obs_rect):
                # // Explosion sound
                self._spawn_particles(robot_rect.center, 'death', 70)
                return -100.0, True

            # Check for clearing an obstacle
            if obs['id'] not in self.cleared_obstacles and obs['x'] + obs['w'] < robot_rect.left:
                # // Clear obstacle sound
                reward += 1.0
                self.cleared_obstacles.add(obs['id'])
        
        return reward, terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_world()
        self._render_particles()
        if not (self.game_over and reward <= -100): # Hide robot on death
             self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance": int(self.distance_traveled),
            "stage": self.stage
        }

    def _render_background(self):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (int(i + self.grid_offset), 0), (int(i + self.grid_offset), self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)
        
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        pygame.draw.line(self.screen, self.COLOR_ROBOT_MAIN, (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 2)

    def _render_world(self):
        # Obstacles
        for obs in self.obstacles:
            rect = (int(obs['x']), int(obs['y']), int(obs['w']), int(obs['h']))
            pygame.gfxdraw.box(self.screen, rect, obs['color'])
            # Outline for pop
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)

        # Finish line
        finish_x = self.FINISH_DISTANCE - self.distance_traveled + self.ROBOT_X_POS
        if 0 < finish_x < self.WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (int(finish_x), 0), (int(finish_x), self.HEIGHT), 5)

    def _render_player(self):
        x, y = int(self.robot_pos[0]), int(self.robot_pos[1])
        
        # Bobbing animation
        bob = math.sin(self.steps * 0.4) * 2 if self.on_ground else 0
        
        # Body
        body_rect = pygame.Rect(x - 12, y - 35 + bob, 24, 25)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT_MAIN, body_rect, border_radius=4)
        
        # Head
        head_pos = (x, y - 30 + bob)
        pygame.draw.circle(self.screen, self.COLOR_ROBOT_MAIN, head_pos, 10)
        
        # Eye
        eye_x_offset = 3
        eye_pos = (x + eye_x_offset, y - 32 + bob)
        pygame.draw.circle(self.screen, self.COLOR_ROBOT_ACCENT, eye_pos, 4)

        # Jetpack feet
        if not self.on_ground:
            # // Jetpack sound
            for i in range(2):
                foot_x = x - 6 + i * 12
                foot_y = y - 10
                pygame.draw.rect(self.screen, self.COLOR_ROBOT_ACCENT, (foot_x, foot_y, 4, 8))
                # Add jet particles
                if self.np_random.random() > 0.3:
                    p = {
                        'x': foot_x + 2, 'y': foot_y + 8,
                        'vx': self.np_random.uniform(-0.5, 0.5), 'vy': self.np_random.uniform(2, 4),
                        'life': self.np_random.integers(5, 15), 'color': random.choice([(255,255,255), self.COLOR_ROBOT_ACCENT])
                    }
                    self.particles.append(p)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life'])))) if 'max_life' in p else 255
            color = p['color'] + (alpha,)
            size = p.get('size', 2) * (p['life'] / p['max_life']) if 'max_life' in p else 2
            pos = (int(p['x']), int(p['y']))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], max(1, int(size)), color)

    def _render_ui(self):
        # Distance
        dist_rem = max(0, self.FINISH_DISTANCE - self.distance_traveled)
        dist_text = f"DISTANCE: {int(dist_rem / 100):03d}m"
        self._draw_text(dist_text, (20, 15), self.font_small)

        # Stage
        stage_text = f"STAGE: {self.stage}"
        self._draw_text(stage_text, (self.WIDTH - 120, 15), self.font_small)
        
        # Game Over / Finish
        if self.game_over:
            if self.distance_traveled >= self.FINISH_DISTANCE:
                msg = "FINISH!"
            else:
                msg = "GAME OVER"
            self._draw_text(msg, (self.WIDTH // 2, self.HEIGHT // 2 - 30), self.font_large, center=True)

    def _draw_text(self, text, pos, font, color=None, shadow=True, center=False):
        if color is None: color = self.COLOR_TEXT
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        if shadow:
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            shadow_rect = shadow_surf.get_rect()
            shadow_rect.topleft = (text_rect.left + 2, text_rect.top + 2)
            if center:
                shadow_rect.center = (pos[0] + 2, pos[1] + 2)
            self.screen.blit(shadow_surf, shadow_rect)
            
        self.screen.blit(text_surf, text_rect)

    def _spawn_particles(self, pos, p_type, count):
        for _ in range(count):
            if p_type == 'jump':
                p = { 'vx': self.np_random.uniform(-1, 1), 'vy': self.np_random.uniform(1, 3), 'color': self.COLOR_ROBOT_ACCENT, 'size': 3}
            elif p_type == 'land':
                angle = self.np_random.uniform(math.pi, 2 * math.pi)
                speed = self.np_random.uniform(1, 3)
                p = { 'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed, 'color': (150, 150, 180), 'size': 4}
            elif p_type == 'death':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 8)
                p = { 'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed, 'color': random.choice(self.OBSTACLE_COLORS + [self.COLOR_ROBOT_MAIN]), 'size': 5}
            else:
                return

            particle = {
                'x': pos[0] + self.np_random.uniform(-5, 5),
                'y': pos[1] + self.np_random.uniform(-5, 5),
                'vx': p['vx'], 'vy': p['vy'],
                'life': self.np_random.integers(20, 40),
                'color': p['color'], 'size': p['size']
            }
            particle['max_life'] = particle['life']
            self.particles.append(particle)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To display the game in a window ---
    pygame.display.set_caption("Robot Runner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        action = env.action_space.sample() # Start with a random action
        action[0] = 0 # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Keyboard controls for human play
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1 # Small Jump
        elif keys[pygame.K_DOWN]:
            action[0] = 2 # Large Jump
        if keys[pygame.K_ESCAPE]:
            running = False
        if keys[pygame.K_r]:
            obs, info = env.reset()
            done = False
            continue

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Blit the environment's screen to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # If done, wait a bit before resetting
        if done:
            pygame.time.wait(1000)
            obs, info = env.reset()
            done = False

    env.close()