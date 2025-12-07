
# Generated: 2025-08-28T05:00:31.675005
# Source Brief: brief_02491.md
# Brief Index: 2491

        
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
        "Controls: ↑↓ to adjust angle in air. Space to perform a somersault for style points. Avoid obstacles!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Side-scrolling ski jump game. Maximize your score by performing stylish tricks, getting airtime, and reaching the bottom of the slope."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # Static variable to track difficulty progression across episodes
    _successful_episodes = 0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game and Screen Dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
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
        
        # Fonts
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_info = pygame.font.SysFont("Consolas", 16)
        
        # Colors
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_MOUNTAIN_BG = (25, 40, 60)
        self.COLOR_SNOW = (240, 245, 255)
        self.COLOR_OBSTACLE = (80, 50, 30)
        self.COLOR_SKIER = (255, 50, 50)
        self.COLOR_SKIER_GLOW = (255, 150, 150)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_TRAJECTORY = (100, 150, 255, 100)
        
        # Physics and Game Constants
        self.GRAVITY = 0.25
        self.MAX_ROTATION_SPEED = 15
        self.ROTATION_INPUT_SPEED = 2.0
        self.FRICTION = 0.99
        self.SLOPE_FRICTION = 0.995
        self.MAX_STEPS = 1500
        self.SLOPE_SEGMENT_LENGTH = 150
        self.SLOPE_SEGMENTS = 80
        self.SLOPE_LENGTH = self.SLOPE_SEGMENT_LENGTH * self.SLOPE_SEGMENTS
        
        # Initialize state variables
        self.skier = {}
        self.slope_points = []
        self.bg_mountains = []
        self.obstacles = []
        self.particles = []
        self.camera_y = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize state by calling reset
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.skier = {
            'pos': pygame.math.Vector2(100, 100),
            'vel': pygame.math.Vector2(2, 0),
            'angle': 0.0,
            'angular_vel': 0.0,
            'on_ground': False,
            'is_trick': False,
            'trick_rotation_total': 0.0,
            'crashed': False,
            'lean': 0.0,
        }
        
        self._generate_slope()
        self._generate_obstacles()
        self._generate_bg_mountains()
        self.particles = []
        self.camera_y = 0

        # Place skier at the start of the slope
        start_y, start_angle = self._get_slope_data(self.skier['pos'].x)
        self.skier['pos'].y = start_y - 10
        self.skier['angle'] = math.degrees(start_angle)
        
        return self._get_observation(), self._get_info()

    def _generate_slope(self):
        self.slope_points = [pygame.math.Vector2(0, 150)]
        current_angle = 0.1
        for i in range(1, self.SLOPE_SEGMENTS + 1):
            x = i * self.SLOPE_SEGMENT_LENGTH
            
            # Introduce variations: ramps, flats, steep sections
            rand_val = self.np_random.random()
            if rand_val < 0.2: # Steepen
                current_angle = min(0.8, current_angle + self.np_random.uniform(0.1, 0.3))
            elif rand_val < 0.4: # Flatten
                current_angle = max(0.05, current_angle - self.np_random.uniform(0.1, 0.3))
            elif rand_val < 0.5: # Jump ramp
                current_angle = -self.np_random.uniform(0.3, 0.5)
            # else, keep similar angle
            
            dy = math.sin(current_angle) * self.SLOPE_SEGMENT_LENGTH
            y = self.slope_points[-1].y + dy
            self.slope_points.append(pygame.math.Vector2(x, y))

    def _generate_obstacles(self):
        self.obstacles = []
        obstacle_count = 3 + int(3 * 0.1 * (GameEnv._successful_episodes // 500))
        
        for _ in range(obstacle_count):
            # Place obstacles on flat or gently sloping parts of the track
            while True:
                segment_idx = self.np_random.integers(5, len(self.slope_points) - 2)
                p1 = self.slope_points[segment_idx]
                p2 = self.slope_points[segment_idx + 1]
                slope_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)

                if -0.2 < slope_angle < 0.5: # Avoid placing on steep ramps
                    break

            pos_on_segment = self.np_random.random()
            x = p1.x + (p2.x - p1.x) * pos_on_segment
            y, _ = self._get_slope_data(x)
            height = self.np_random.integers(20, 50)
            width = self.np_random.integers(15, 30)
            self.obstacles.append(pygame.Rect(x - width / 2, y - height, width, height))

    def _generate_bg_mountains(self):
        self.bg_mountains = []
        for i in range(20):
            x = self.np_random.uniform(-self.WIDTH, self.SLOPE_LENGTH + self.WIDTH)
            y = self.np_random.uniform(0, self.SLOPE_LENGTH)
            scale = self.np_random.uniform(0.5, 2.0)
            points = [
                (x, y),
                (x + 100 * scale, y - 200 * scale),
                (x + 200 * scale, y)
            ]
            self.bg_mountains.append(points)
            
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        self._handle_input(movement, space_held)
        self._update_physics()
        self._check_collisions_and_state()
        self._update_particles()
        
        self.camera_y = self.skier['pos'].y - self.HEIGHT / 2
        self.steps += 1
        
        reward = self._calculate_reward()
        terminated = self.game_over or self.steps >= self.MAX_STEPS

        if terminated and not self.skier['crashed']:
            # Successful finish
            if self.skier['pos'].x >= self.SLOPE_LENGTH - self.WIDTH:
                 self.score += 100
                 reward = 100.0
                 GameEnv._successful_episodes += 1
            # Timed out
            else:
                 reward = 0.0
        elif terminated and self.skier['crashed']:
            self.score -= 100
            reward = -100.0
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        skier = self.skier
        # Cosmetic lean
        if movement == 3: skier['lean'] = -1
        elif movement == 4: skier['lean'] = 1
        else: skier['lean'] = 0

        if not skier['on_ground']:
            # Mid-air rotation control
            if movement == 1:  # Up
                skier['angular_vel'] -= self.ROTATION_INPUT_SPEED
            elif movement == 2:  # Down
                skier['angular_vel'] += self.ROTATION_INPUT_SPEED
            
            # Initiate trick
            if space_held and not skier['is_trick']:
                skier['is_trick'] = True
                skier['angular_vel'] = self.MAX_ROTATION_SPEED * (-1 if skier['angle'] < 90 else 1)
                # sfx: whoosh_start_trick
        else:
            # On ground, space does nothing, rotation is locked to slope
            skier['angular_vel'] = 0

    def _update_physics(self):
        skier = self.skier
        
        if skier['on_ground']:
            slope_y, slope_angle_rad = self._get_slope_data(skier['pos'].x)
            
            # Force parallel to slope
            force = self.GRAVITY * math.sin(slope_angle_rad)
            skier['vel'].x += force * math.cos(slope_angle_rad)
            skier['vel'].y += force * math.sin(slope_angle_rad)
            
            # Apply friction
            skier['vel'] *= self.SLOPE_FRICTION
            
            # Stick to slope
            skier['pos'] += skier['vel']
            skier['pos'].y = slope_y - 5 # 5 is half skier height
            
            # Align angle to slope
            target_angle = math.degrees(slope_angle_rad)
            skier['angle'] = skier['angle'] * 0.8 + target_angle * 0.2

            # Check for takeoff
            if skier['pos'].y < slope_y - 6:
                skier['on_ground'] = False
        else: # In air
            skier['vel'].y += self.GRAVITY
            skier['vel'] *= self.FRICTION
            skier['pos'] += skier['vel']
            
            # Update angle
            skier['angular_vel'] = max(-self.MAX_ROTATION_SPEED, min(self.MAX_ROTATION_SPEED, skier['angular_vel']))
            skier['angle'] += skier['angular_vel']
            skier['angular_vel'] *= 0.95 # Dampen rotation

            if skier['is_trick']:
                skier['trick_rotation_total'] += skier['angular_vel']
                
            skier['angle'] %= 360

    def _check_collisions_and_state(self):
        skier = self.skier
        slope_y, slope_angle_rad = self._get_slope_data(skier['pos'].x)

        # Landing
        if not skier['on_ground'] and skier['pos'].y >= slope_y - 5:
            skier['on_ground'] = True
            skier['pos'].y = slope_y - 5
            
            # sfx: land_snow
            self._create_particles(skier['pos'], 30, skier['vel'].length() * 2)

            # Check landing angle
            skier_angle_rad = math.radians(skier['angle'])
            angle_diff = abs(self._normalize_angle(skier_angle_rad) - self._normalize_angle(slope_angle_rad))
            
            if angle_diff > 0.8: # ~45 degrees tolerance
                self._crash()
                return

            # Check trick completion
            if skier['is_trick']:
                # Completed at least one full rotation?
                if abs(skier['trick_rotation_total']) > 300:
                    self.score += 50 # Trick bonus
                    # sfx: success_chime
                skier['is_trick'] = False
                skier['trick_rotation_total'] = 0

            # Adjust velocity based on landing
            skier['vel'] = skier['vel'].reflect(pygame.math.Vector2(math.cos(slope_angle_rad), math.sin(slope_angle_rad)).rotate(90))
            skier['vel'] *= 0.7 # Landing speed loss
            
        # Obstacle collision
        skier_rect = pygame.Rect(skier['pos'].x - 5, skier['pos'].y - 10, 10, 20)
        for obs in self.obstacles:
            if obs.colliderect(skier_rect):
                self._crash()
                return

        # Out of bounds / Finish line
        if skier['pos'].x > self.SLOPE_LENGTH or skier['pos'].y > self.slope_points[-1].y + self.HEIGHT:
            self.game_over = True

    def _crash(self):
        self.skier['crashed'] = True
        self.game_over = True
        # sfx: crash_sound
        self._create_particles(self.skier['pos'], 100, 15, life=80)

    def _get_slope_data(self, x_pos):
        if x_pos <= 0: return self.slope_points[0].y, 0.1
        if x_pos >= self.slope_points[-1].x: return self.slope_points[-1].y, 0.1

        # Find segment
        p1 = self.slope_points[int(x_pos // self.SLOPE_SEGMENT_LENGTH)]
        p2 = self.slope_points[int(x_pos // self.SLOPE_SEGMENT_LENGTH) + 1]
        
        # Interpolate
        t = (x_pos - p1.x) / (p2.x - p1.x) if (p2.x - p1.x) != 0 else 0
        y = p1.y + t * (p2.y - p1.y)
        angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
        return y, angle

    @staticmethod
    def _normalize_angle(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _calculate_reward(self):
        if self.skier['crashed']: return 0 # Terminal reward is handled in step()
        
        reward = 0.1  # Survival reward
        
        # Airtime reward
        if not self.skier['on_ground']:
            reward += 0.2
            
        # Proximity to obstacle penalty
        min_dist = float('inf')
        for obs in self.obstacles:
            if obs.x > self.skier['pos'].x:
                dist = pygame.math.Vector2(obs.centerx, obs.centery).distance_to(self.skier['pos'])
                if dist < min_dist:
                    min_dist = dist
        
        if min_dist < 200:
            reward -= 0.5 * (1 - min_dist / 200)

        return reward

    def _create_particles(self, pos, count, speed, life=40):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * self.np_random.uniform(0.5, 1.5) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(life // 2, life),
                'size': self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.9
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Background Elements (Parallax) ---
        cam_x_bg = self.skier['pos'].x * 0.5
        cam_y_bg = self.camera_y * 0.5
        for mountain in self.bg_mountains:
            points_on_screen = [(p[0] - cam_x_bg, p[1] - cam_y_bg) for p in mountain]
            pygame.gfxdraw.aapolygon(self.screen, points_on_screen, self.COLOR_MOUNTAIN_BG)
            pygame.gfxdraw.filled_polygon(self.screen, points_on_screen, self.COLOR_MOUNTAIN_BG)

        # --- Draw Slope ---
        world_to_screen = lambda p: (int(p.x - self.skier['pos'].x + self.WIDTH/2), int(p.y - self.camera_y))
        
        for i in range(len(self.slope_points) - 1):
            p1 = self.slope_points[i]
            p2 = self.slope_points[i+1]
            
            # Culling
            if p2.y < self.camera_y - 20 or p1.y > self.camera_y + self.HEIGHT + 20:
                continue

            sp1 = world_to_screen(p1)
            sp2 = world_to_screen(p2)
            
            # Draw the snow polygon
            poly_points = [sp1, sp2, (sp2[0], sp2[1] + 20), (sp1[0], sp1[1] + 20)]
            pygame.gfxdraw.filled_polygon(self.screen, poly_points, self.COLOR_SNOW)
            pygame.draw.aaline(self.screen, self.COLOR_SNOW, sp1, sp2)

        # --- Draw Obstacles ---
        for obs in self.obstacles:
            if obs.bottom < self.camera_y or obs.top > self.camera_y + self.HEIGHT:
                continue
            screen_rect = obs.move(-self.skier['pos'].x + self.WIDTH/2, -self.camera_y)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)

        # --- Draw Particles ---
        for p in self.particles:
            screen_pos = (int(p['pos'].x - self.skier['pos'].x + self.WIDTH/2), int(p['pos'].y - self.camera_y))
            size = int(p['size'] * (p['life'] / 40.0))
            if size > 0:
                pygame.draw.circle(self.screen, self.COLOR_SNOW, screen_pos, size)
                
        # --- Draw Trajectory Prediction ---
        if not self.skier['on_ground'] and not self.skier['is_trick']:
            pos = self.skier['pos'].copy()
            vel = self.skier['vel'].copy()
            for i in range(30):
                vel.y += self.GRAVITY
                vel *= self.FRICTION
                pos += vel
                if i % 3 == 0:
                    screen_pos = world_to_screen(pos)
                    pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], 2, self.COLOR_TRAJECTORY)

        # --- Draw Skier ---
        if not self.skier['crashed']:
            self._draw_skier()

    def _draw_skier(self):
        skier = self.skier
        screen_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        
        # Create a surface for the skier to rotate
        skier_surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        center = (20, 20)
        
        # Body
        body_start = (center[0], center[1] - 8)
        body_end = (center[0], center[1] + 8)
        pygame.draw.line(skier_surf, self.COLOR_SKIER, body_start, body_end, 6)
        
        # Head
        pygame.gfxdraw.filled_circle(skier_surf, center[0], center[1] - 10, 5, self.COLOR_SKIER)
        pygame.gfxdraw.aacircle(skier_surf, center[0], center[1] - 10, 5, self.COLOR_SKIER)
        
        # Skis with cosmetic lean
        lean_offset = skier['lean'] * 4
        ski_start = (center[0] - 12 + lean_offset, center[1] + 8)
        ski_end = (center[0] + 12 + lean_offset, center[1] + 8)
        pygame.draw.line(skier_surf, self.COLOR_SKIER, ski_start, ski_end, 3)

        # Rotate and blit
        rotated_surf = pygame.transform.rotate(skier_surf, -skier['angle'])
        new_rect = rotated_surf.get_rect(center=screen_pos)
        
        # Glow effect
        glow_surf = pygame.transform.scale(rotated_surf, (new_rect.width + 10, new_rect.height + 10))
        glow_surf.fill(self.COLOR_SKIER_GLOW, special_flags=pygame.BLEND_RGB_ADD)
        self.screen.blit(glow_surf, glow_surf.get_rect(center=screen_pos))
        
        self.screen.blit(rotated_surf, new_rect.topleft)

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        speed_text = self.font_info.render(f"Speed: {self.skier['vel'].length():.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_text, (10, 40))
        
        angle_text = self.font_info.render(f"Angle: {int(self.skier['angle'])}°", True, self.COLOR_UI_TEXT)
        self.screen.blit(angle_text, (10, 60))

        if self.skier['is_trick']:
            trick_text = self.font_main.render("TRICK!", True, self.COLOR_SKIER_GLOW)
            self.screen.blit(trick_text, (self.WIDTH // 2 - trick_text.get_width() // 2, 20))

        if self.game_over:
            status = "CRASHED!" if self.skier['crashed'] else "FINISHED!"
            if self.steps >= self.MAX_STEPS and not self.skier['crashed']:
                status = "TIME'S UP!"
            
            end_text = self.font_main.render(status, True, self.COLOR_UI_TEXT)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - 50))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "skier_pos": (self.skier['pos'].x, self.skier['pos'].y),
            "skier_vel": (self.skier['vel'].x, self.skier['vel'].y),
            "on_ground": self.skier['on_ground'],
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
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Ski Game")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}, Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting
            
        clock.tick(30) # Match the environment's internal clock
        
    env.close()