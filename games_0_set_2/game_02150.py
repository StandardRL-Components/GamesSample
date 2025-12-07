
# Generated: 2025-08-27T19:25:25.918867
# Source Brief: brief_02150.md
# Brief Index: 2150

        
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
        "Controls: ↑ to accelerate, ↓ to brake/reverse, ←→ to turn. "
        "Hold Shift to drift and press Space to boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down arcade racer. Drift through corners on a "
        "procedurally generated track, manage your boost, and race against "
        "three opponents to finish first."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)
        
        # Colors
        self.COLOR_BG = (30, 40, 50)
        self.COLOR_TRACK = (100, 100, 110)
        self.COLOR_LINES = (200, 200, 220)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_OPPONENTS = [(50, 150, 255), (255, 255, 50), (50, 255, 150)]
        self.COLOR_START_FINISH = (50, 255, 50)
        self.COLOR_BOOST = (255, 150, 0)
        self.COLOR_SKID = (20, 20, 20)
        self.COLOR_UI_TEXT = (240, 240, 240)
        
        # Game constants
        self.NUM_OPPONENTS = 3
        self.TOTAL_LAPS = 3
        self.MAX_STEPS = 5000
        self.TRACK_WIDTH = 80
        self.KART_WIDTH, self.KART_LENGTH = 12, 22

        # Physics parameters
        self.ACCELERATION = 0.15
        self.BRAKING = 0.3
        self.FRICTION = 0.04
        self.TURN_SPEED = 0.05
        self.DRIFT_TURN_MULT = 1.8
        self.DRIFT_FRICTION_MULT = 0.5
        self.MAX_SPEED = 5.0
        self.MAX_REVERSE_SPEED = -2.0
        self.BOOST_POWER = 0.4
        self.BOOST_DRAIN = 4.0
        self.BOOST_REGEN = 0.5
        self.WALL_PENALTY = 0.5

        # Initialize state variables
        self.track_centerline = []
        self.track_border1 = []
        self.track_border2 = []
        self.checkpoints = []
        self.player = {}
        self.opponents = []
        self.particles = []
        self.skid_marks = []
        self.steps = 0
        self.game_over = False

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.game_over = False
        self.particles.clear()
        self.skid_marks.clear()
        
        self._generate_track()
        self._reset_karts()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        self.steps += 1
        
        prev_ranks = self._get_ranks()
        
        # Update game objects
        self._update_player(movement, space_held, shift_held)
        self._update_opponents()
        self._update_particles()
        
        # Handle state changes and rewards
        reward = self._calculate_rewards(prev_ranks)
        
        # Check for termination
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            reward += self._get_terminal_reward()

        # Enforce max steps
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            if self._get_player_rank() == 4:
                 reward -= 100 # Penalty for running out of time
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_track(self):
        self.track_centerline.clear()
        num_points = 12
        center_x, center_y = 0, 0
        min_radius, max_radius = 250, 450

        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            radius = self.np_random.uniform(min_radius, max_radius)
            noise_x = self.np_random.uniform(-50, 50)
            noise_y = self.np_random.uniform(-50, 50)
            x = center_x + radius * math.cos(angle) + noise_x
            y = center_y + radius * math.sin(angle) + noise_y
            points.append((x, y))

        # Create a smooth-ish loop using Catmull-Rom spline logic
        for i in range(num_points):
            p0 = points[(i - 1 + num_points) % num_points]
            p1 = points[i]
            p2 = points[(i + 1) % num_points]
            p3 = points[(i + 2) % num_points]
            for t in np.linspace(0, 1, 20, endpoint=False):
                x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t**2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t**3)
                y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t**2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t**3)
                self.track_centerline.append(np.array([x, y]))
        
        self.checkpoints = self.track_centerline[::len(self.track_centerline) // 16]

    def _reset_karts(self):
        start_pos = self.track_centerline[0]
        start_angle = self._get_path_angle(0)
        
        perp_angle = start_angle + math.pi / 2
        perp_vec = np.array([math.cos(perp_angle), math.sin(perp_angle)])

        positions = [
            start_pos - perp_vec * self.TRACK_WIDTH * 0.3,
            start_pos + perp_vec * self.TRACK_WIDTH * 0.3,
            start_pos - perp_vec * self.TRACK_WIDTH * 0.3 - np.array([math.cos(start_angle), math.sin(start_angle)]) * 40,
            start_pos + perp_vec * self.TRACK_WIDTH * 0.3 - np.array([math.cos(start_angle), math.sin(start_angle)]) * 40,
        ]
        
        self.player = self._create_kart(positions[0], start_angle, self.COLOR_PLAYER)
        self.player['boost'] = 100.0

        self.opponents.clear()
        for i in range(self.NUM_OPPONENTS):
            kart = self._create_kart(positions[i+1], start_angle, self.COLOR_OPPONENTS[i])
            kart['speed_mult'] = self.np_random.uniform(0.92, 0.98)
            kart['turn_ability'] = self.np_random.uniform(0.8, 1.0)
            self.opponents.append(kart)

    def _create_kart(self, pos, angle, color):
        return {
            "pos": np.array(pos, dtype=float),
            "vel": np.array([0.0, 0.0]),
            "angle": angle,
            "speed": 0.0,
            "color": color,
            "lap": 0,
            "next_checkpoint": 1,
            "dist_to_center": 0.0,
            "is_drifting": False,
        }

    def _update_player(self, movement, space_held, shift_held):
        kart = self.player
        
        # Boost
        if space_held and kart['boost'] > 0:
            # sfx: boost
            accel_mod = self.BOOST_POWER
            kart['boost'] -= self.BOOST_DRAIN
            self._add_particles(kart['pos'], kart['angle'], num=3, color=self.COLOR_BOOST)
        else:
            accel_mod = 0
            kart['boost'] = min(100.0, kart['boost'] + self.BOOST_REGEN)

        # Acceleration/Braking
        if movement == 1: # Up
            kart['speed'] += self.ACCELERATION + accel_mod
        elif movement == 2: # Down
            kart['speed'] -= self.BRAKING
        
        # Steering
        steer_input = 0
        if movement == 3: # Left
            steer_input = -1
        elif movement == 4: # Right
            steer_input = 1

        # Drift mechanics
        kart['is_drifting'] = shift_held and abs(kart['speed']) > 1.0 and steer_input != 0
        turn_mod = self.DRIFT_TURN_MULT if kart['is_drifting'] else 1.0
        friction_mod = self.DRIFT_FRICTION_MULT if kart['is_drifting'] else 1.0
        
        if kart['is_drifting']:
            # sfx: skid
            self._add_skid_mark(kart['pos'])

        # Apply physics
        kart['angle'] += steer_input * self.TURN_SPEED * turn_mod * (kart['speed'] / self.MAX_SPEED)
        kart['speed'] *= (1.0 - self.FRICTION * friction_mod)
        kart['speed'] = np.clip(kart['speed'], self.MAX_REVERSE_SPEED, self.MAX_SPEED)
        
        # Update velocity and position
        forward_vec = np.array([math.cos(kart['angle']), math.sin(kart['angle'])])
        kart['vel'] = forward_vec * kart['speed']
        kart['pos'] += kart['vel']

    def _update_opponents(self):
        for kart in self.opponents:
            target_point = self.checkpoints[kart['next_checkpoint'] % len(self.checkpoints)]
            
            # Add some imperfection to the target
            target_point = target_point + self.np_random.uniform(-5, 5, size=2)

            # Basic AI: steer towards the target checkpoint
            angle_to_target = math.atan2(target_point[1] - kart['pos'][1], target_point[0] - kart['pos'][0])
            angle_diff = (angle_to_target - kart['angle'] + math.pi) % (2 * math.pi) - math.pi
            
            turn_dir = np.clip(angle_diff * 4, -1, 1) * kart['turn_ability']
            kart['angle'] += turn_dir * self.TURN_SPEED * (kart['speed'] / self.MAX_SPEED) * 0.8
            
            # Maintain speed
            kart['speed'] = max(2.0, kart['speed'] + self.ACCELERATION)
            kart['speed'] *= (1.0 - self.FRICTION)
            kart['speed'] = min(kart['speed'], self.MAX_SPEED * kart['speed_mult'])
            
            forward_vec = np.array([math.cos(kart['angle']), math.sin(kart['angle'])])
            kart['vel'] = forward_vec * kart['speed']
            kart['pos'] += kart['vel']

    def _calculate_rewards(self, prev_ranks):
        reward = 0
        all_karts = [self.player] + self.opponents

        for i, kart in enumerate(all_karts):
            # Wall collision
            min_dist = float('inf')
            closest_point_idx = 0
            for j, p in enumerate(self.track_centerline):
                dist = np.linalg.norm(kart['pos'] - p)
                if dist < min_dist:
                    min_dist = dist
                    closest_point_idx = j
            
            kart['dist_to_center'] = min_dist
            
            if min_dist > self.TRACK_WIDTH / 2:
                # sfx: collision
                kart['speed'] *= (1.0 - self.WALL_PENALTY)
                # Push back towards track
                direction_to_center = self.track_centerline[closest_point_idx] - kart['pos']
                kart['pos'] += direction_to_center * 0.1

                if i == 0: # Player
                    reward -= 0.1
            
            # Checkpoint and Lap logic
            dist_to_checkpoint = np.linalg.norm(kart['pos'] - self.checkpoints[kart['next_checkpoint'] % len(self.checkpoints)])
            if dist_to_checkpoint < self.TRACK_WIDTH * 0.75:
                kart['next_checkpoint'] += 1
                if kart['next_checkpoint']-1 == len(self.checkpoints):
                    kart['lap'] += 1
                    kart['next_checkpoint'] = 1
                    if i == 0: # Player
                        # sfx: lap_complete
                        reward += 5.0
        
        # Overtake reward
        current_ranks = self._get_ranks()
        player_prev_rank = prev_ranks.index(0)
        player_current_rank = current_ranks.index(0)
        
        if player_current_rank < player_prev_rank:
            # sfx: overtake
            reward += 1.0 * (player_prev_rank - player_current_rank)

        # Reward for speed
        reward += (self.player['speed'] / self.MAX_SPEED) * 0.01

        return reward

    def _get_ranks(self):
        kart_progress = []
        for i, kart in enumerate([self.player] + self.opponents):
            dist_to_next_cp = np.linalg.norm(kart['pos'] - self.checkpoints[kart['next_checkpoint'] % len(self.checkpoints)])
            progress = (kart['lap'] * len(self.checkpoints)) + kart['next_checkpoint'] - (dist_to_next_cp / 10000.0)
            kart_progress.append((-progress, i)) # Negate for descending sort
        
        kart_progress.sort()
        return [item[1] for item in kart_progress]
    
    def _get_player_rank(self):
        ranks = self._get_ranks()
        return ranks.index(0) + 1

    def _get_terminal_reward(self):
        rank = self._get_player_rank()
        if rank == 1: return 50.0
        if rank == 2: return 25.0
        if rank == 3: return 10.0
        return -100.0

    def _check_termination(self):
        for kart in [self.player] + self.opponents:
            if kart['lap'] >= self.TOTAL_LAPS:
                return True
        return False

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _add_particles(self, pos, angle, num, color):
        for _ in range(num):
            vel_angle = angle + math.pi + self.np_random.uniform(-0.5, 0.5)
            vel_mag = self.np_random.uniform(1, 3)
            vel = np.array([math.cos(vel_angle), math.sin(vel_angle)]) * vel_mag
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'color': color,
            })
    
    def _add_skid_mark(self, pos):
        if not self.skid_marks or np.linalg.norm(pos - self.skid_marks[-1]['pos']) > 5:
            self.skid_marks.append({
                'pos': pos.copy(),
                'life': 150 # lasts 5 seconds at 30fps
            })
        self.skid_marks = [s for s in self.skid_marks if s['life'] > 0]
        for s in self.skid_marks:
            s['life'] -= 1
            
    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "player_lap": self.player['lap'],
            "player_rank": self._get_player_rank() if not self.game_over else self.info_cache['player_rank'],
            "player_boost": self.player['boost'],
            "steps": self.steps,
        }
    
    def _render_all(self):
        # Center camera on player
        cam_offset = -self.player['pos'] + np.array([self.WIDTH / 2, self.HEIGHT / 2])
        cam_angle = -self.player['angle']

        self.screen.fill(self.COLOR_BG)
        
        self._render_track(cam_offset, cam_angle)
        self._render_effects(cam_offset, cam_angle)
        self._render_karts(cam_offset, cam_angle)
        self._render_ui()

        if self.game_over:
            self._render_game_over()

    def _render_track(self, offset, angle):
        # Render track surface
        screen_points = [self._transform_point(p, offset, angle) for p in self.track_centerline]
        pygame.draw.lines(self.screen, self.COLOR_TRACK, True, screen_points, self.TRACK_WIDTH)
        pygame.draw.lines(self.screen, self.COLOR_LINES, True, screen_points, 2)
        
        # Render start/finish line
        p1 = self.track_centerline[0]
        p2 = self.track_centerline[-1]
        mid = (p1 + p2) / 2
        line_angle = self._get_path_angle(0) + math.pi / 2
        line_vec = np.array([math.cos(line_angle), math.sin(line_angle)]) * self.TRACK_WIDTH / 2
        
        start = self._transform_point(mid - line_vec, offset, angle)
        end = self._transform_point(mid + line_vec, offset, angle)
        pygame.draw.line(self.screen, self.COLOR_START_FINISH, start, end, 5)

    def _render_effects(self, offset, angle):
        # Render skid marks
        for skid in self.skid_marks:
            alpha = max(0, min(255, int(skid['life'] * 2)))
            color = (*self.COLOR_SKID, alpha)
            pos = self._transform_point(skid['pos'], offset, angle)
            temp_surf = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (3, 3), 3)
            self.screen.blit(temp_surf, (pos[0] - 3, pos[1] - 3))
            
        # Render particles
        for p in self.particles:
            pos = self._transform_point(p['pos'], offset, angle)
            radius = int(p['life'] / 4)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, p['color'])

    def _render_karts(self, offset, angle):
        karts_to_render = self.opponents + [self.player]
        for i, kart in enumerate(karts_to_render):
            is_player = (i == len(karts_to_render) - 1)
            
            kart_angle = kart['angle'] + angle
            
            # For player, position is always center. For others, transform.
            if is_player:
                screen_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2])
                # Player glow
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), self.KART_LENGTH // 2 + 5, (*self.COLOR_PLAYER, 80))

            else:
                screen_pos = self._transform_point(kart['pos'], offset, angle)
            
            # Create kart polygon
            hw, hl = self.KART_WIDTH / 2, self.KART_LENGTH / 2
            points = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
            rotated_points = [self._rotate_point(p, kart_angle) for p in points]
            screen_points = [(p[0] + screen_pos[0], p[1] + screen_pos[1]) for p in rotated_points]
            
            pygame.gfxdraw.aapolygon(self.screen, screen_points, kart['color'])
            pygame.gfxdraw.filled_polygon(self.screen, screen_points, kart['color'])

    def _render_ui(self):
        # Lap Counter
        lap_text = f"LAP: {min(self.TOTAL_LAPS, self.player['lap'] + 1)} / {self.TOTAL_LAPS}"
        text_surf = self.font_small.render(lap_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Position
        rank = self._get_player_rank()
        rank_str = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[rank]
        pos_text = f"POS: {rank_str}"
        text_surf = self.font_small.render(pos_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))

        # Speed
        speed_kmh = int(abs(self.player['speed']) * 30)
        speed_text = f"{speed_kmh} KM/H"
        text_surf = self.font_small.render(speed_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, self.HEIGHT - 30))

        # Boost Gauge
        boost_rect = pygame.Rect(10, self.HEIGHT - 30, 150, 20)
        pygame.draw.rect(self.screen, (50,50,50), boost_rect)
        fill_width = (self.player['boost'] / 100.0) * 150
        fill_rect = pygame.Rect(10, self.HEIGHT - 30, fill_width, 20)
        pygame.draw.rect(self.screen, self.COLOR_BOOST, fill_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, boost_rect, 2)
        
    def _render_game_over(self):
        # Cache info at the moment of game over
        if 'player_rank' not in self.info_cache:
            self.info_cache = self._get_info()

        rank = self.info_cache['player_rank']
        rank_str = {1: "1st PLACE!", 2: "2nd PLACE", 3: "3rd PLACE", 4: "4th PLACE"}[rank]
        
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        
        text_surf = self.font_large.render(rank_str, True, self.COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        
        overlay.blit(text_surf, text_rect)
        self.screen.blit(overlay, (0, 0))

    def _get_path_angle(self, idx):
        p1 = self.track_centerline[idx % len(self.track_centerline)]
        p2 = self.track_centerline[(idx + 1) % len(self.track_centerline)]
        return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

    def _transform_point(self, pos, offset, angle):
        translated = pos + offset
        rotated = self._rotate_point(translated - np.array([self.WIDTH/2, self.HEIGHT/2]), angle)
        return rotated + np.array([self.WIDTH/2, self.HEIGHT/2])

    @staticmethod
    def _rotate_point(p, angle, center=(0,0)):
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        px, py = p[0] - center[0], p[1] - center[1]
        return (px * cos_a - py * sin_a + center[0], px * sin_a + py * cos_a + center[1])

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Arcade Racer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    done = False
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    # Game loop for human play
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Map keyboard to MultiDiscrete action space ---
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # None
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        # Space button
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift button
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over. Final Info: {info}")
            # Allow viewing the final screen for a moment
            for _ in range(90): # 3 seconds at 30fps
                screen.blit(env.screen, (0, 0))
                pygame.display.flip()
                env.clock.tick(30)
            done = True

        # Render the observation to the display window
        screen.blit(env.screen, (0, 0))
        pygame.display.flip()

        # Control the frame rate
        env.clock.tick(30)
        
    env.close()