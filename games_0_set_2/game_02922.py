# Generated: 2025-08-27T21:50:56.748850
# Source Brief: brief_02922.md
# Brief Index: 2922

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    user_guide = (
        "Controls: ↑ to accelerate, ↓ to brake/reverse, ←→ to turn. "
        "Hold Shift to drift. Press Space to use a collected speed boost."
    )

    game_description = (
        "A fast-paced, top-down arcade racer. Navigate a procedurally generated track, "
        "dodge obstacles, and use speed boosts to get the best time."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_TRACK = (100, 100, 110)
    COLOR_TRACK_BORDER = (180, 180, 190)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_PLAYER_GLOW = (255, 100, 100, 50)
    COLOR_OBSTACLE = (50, 150, 255)
    COLOR_BOOST = (255, 220, 0)
    COLOR_CHECKPOINT = (50, 255, 100)
    COLOR_FINISH_LIGHT = (220, 220, 220)
    COLOR_FINISH_DARK = (50, 50, 50)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_PARTICLE_CRASH = (255, 80, 80)
    COLOR_PARTICLE_DRIFT = (200, 200, 200)

    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game
    FPS = 30
    TIME_LIMIT_SECONDS = 60
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS
    MAX_COLLISIONS = 3
    TRACK_WIDTH = 120
    OBSTACLE_SPAWN_RATE_INITIAL = 0.05
    OBSTACLE_SPAWN_RATE_INCREASE = 0.005 # Per second

    # Player Physics
    ACCELERATION = 0.4
    BRAKE_FORCE = 0.8
    MAX_SPEED = 8.0
    TURN_SPEED = 0.08  # Radians per frame
    FRICTION = 0.96
    DRIFT_FRICTION = 0.92
    DRIFT_CONTROL = 0.7
    BOOST_SPEED_MULTIPLIER = 2.0
    BOOST_DURATION = 2.0 # seconds
    BOUNDARY_BOUNCE = -0.5

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        self.render_mode = render_mode
        self.np_random = None
        
        # Will be initialized in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_vel = None
        self.player_angle = 0.0
        self.collision_count = 0
        self.time_elapsed = 0.0
        self.no_progress_timer = 0
        self.boosts_available = 0
        self.boost_active_timer = 0.0
        self.track_centerline = []
        self.checkpoints = []
        self.current_checkpoint_index = 0
        self.obstacles = []
        self.boosts = []
        self.particles = []
        self.camera_shake = 0
        self.last_dist_to_checkpoint = 0
        self._last_collision_count = 0
        self._last_boosts_available = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = 0.0
        
        self.collision_count = 0
        self.time_elapsed = 0.0
        self.no_progress_timer = 0
        
        self.boosts_available = 0
        self.boost_active_timer = 0.0
        
        self._generate_track()
        self._populate_track()
        
        self.particles = []
        self.camera_shake = 0
        
        self.last_dist_to_checkpoint = self._get_dist_to_next_checkpoint()
        self._last_collision_count = 0
        self._last_boosts_available = 0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.auto_advance and self.render_mode != "human":
            self.clock.tick(self.FPS)

        if not self.game_over:
            self._update_player(action)
            self._update_world()
            self._check_collisions()

        self.steps += 1
        self.time_elapsed += 1 / self.FPS
        
        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Turning ---
        turn_input = 0
        if movement == 3:  # Left
            turn_input = -1
        elif movement == 4: # Right
            turn_input = 1
        
        speed_factor = 1.0 - (self.player_vel.length() / self.MAX_SPEED) * 0.5
        if shift_held:
            speed_factor = 1.0

        self.player_angle += turn_input * self.TURN_SPEED * speed_factor

        # --- Acceleration ---
        acceleration_input = 0
        if movement == 1:  # Up
            acceleration_input = 1
        elif movement == 2: # Down
            acceleration_input = -1

        # --- Boost ---
        if space_held and self.boosts_available > 0 and self.boost_active_timer <= 0:
            self.boosts_available -= 1
            self.boost_active_timer = self.BOOST_DURATION
        
        current_max_speed = self.MAX_SPEED
        if self.boost_active_timer > 0:
            current_max_speed *= self.BOOST_SPEED_MULTIPLIER
            self.boost_active_timer -= 1 / self.FPS
            self._create_particles(self.player_pos, 2, (255, 255, 150), 0.5, speed=2, angle_offset=math.pi, angle_spread=0.5)

        # --- Physics ---
        acc_vec = pygame.math.Vector2(0, 0)
        if acceleration_input > 0:
            acc_vec.from_polar((self.ACCELERATION, -math.degrees(self.player_angle)))
        elif acceleration_input < 0:
            acc_vec.from_polar((self.BRAKE_FORCE, -math.degrees(self.player_angle + math.pi)))

        self.player_vel += acc_vec
        
        # --- Friction & Drifting ---
        if shift_held:
            forward_velocity = self.player_vel.dot(pygame.math.Vector2(1, 0).rotate(-math.degrees(self.player_angle)))
            forward_vec = pygame.math.Vector2(forward_velocity, 0).rotate(-math.degrees(self.player_angle))
            sideways_vec = self.player_vel - forward_vec
            
            forward_vec *= self.FRICTION
            sideways_vec *= self.DRIFT_FRICTION
            self.player_vel = forward_vec + sideways_vec

            if self.player_vel.length() > 2.0:
                 self._create_particles(self.player_pos, 1, self.COLOR_PARTICLE_DRIFT, 0.3, speed=1)
        else:
            self.player_vel *= self.FRICTION

        if self.player_vel.length() > current_max_speed:
            self.player_vel.scale_to_length(current_max_speed)

        self.player_pos += self.player_vel

    def _update_world(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1 / self.FPS
        
        if self.camera_shake > 0:
            self.camera_shake -= 1

        spawn_chance = (self.OBSTACLE_SPAWN_RATE_INITIAL + self.time_elapsed * self.OBSTACLE_SPAWN_RATE_INCREASE) / self.FPS
        if self.np_random.random() < spawn_chance:
            self._spawn_obstacle()

    def _check_collisions(self):
        if self.player_pos is None: return
        
        for obs in self.obstacles[:]:
            dist = self.player_pos.distance_to(obs['pos'])
            if dist < 15 + obs['radius']:
                self.obstacles.remove(obs)
                self.collision_count += 1
                self.player_vel *= 0.5
                self.camera_shake = 15
                self._create_particles(self.player_pos, 20, self.COLOR_PARTICLE_CRASH, 1.0, speed=4)
                break

        for boost in self.boosts[:]:
            dist = self.player_pos.distance_to(boost['pos'])
            if dist < 15 + 10:
                self.boosts.remove(boost)
                self.boosts_available += 1
                break
        
        if self.current_checkpoint_index < len(self.checkpoints):
            checkpoint_pos = self.checkpoints[self.current_checkpoint_index]
            if self.player_pos.distance_to(checkpoint_pos) < self.TRACK_WIDTH / 2:
                self.current_checkpoint_index += 1
                self.no_progress_timer = 0

        p1, p2, _ = self._get_closest_track_segment()
        if p1 is None: return

        player_on_segment_proj = (self.player_pos - p1).dot(p2 - p1) / (p2 - p1).length_squared()
        clamped_proj = max(0, min(1, player_on_segment_proj))
        closest_point_on_line = p1 + (p2 - p1) * clamped_proj
        
        dist_to_centerline = self.player_pos.distance_to(closest_point_on_line)
        if dist_to_centerline > self.TRACK_WIDTH / 2:
            normal = (self.player_pos - closest_point_on_line).normalize()
            self.player_pos = closest_point_on_line + normal * (self.TRACK_WIDTH / 2)
            self.player_vel = self.player_vel.reflect(normal) * abs(self.BOUNDARY_BOUNCE)

    def _calculate_reward(self):
        reward = 0.0

        current_dist = self._get_dist_to_next_checkpoint()
        progress = self.last_dist_to_checkpoint - current_dist
        reward += progress * 0.01
        self.last_dist_to_checkpoint = current_dist

        if progress <= 0:
            self.no_progress_timer += 1
        else:
            self.no_progress_timer = 0
        
        if self.no_progress_timer > self.FPS * 1.5:
            reward -= 0.5
            self.no_progress_timer = 0

        if self.collision_count > self._last_collision_count:
            reward -= 10.0
        self._last_collision_count = self.collision_count

        if self.boosts_available > self._last_boosts_available:
            reward += 5.0
        self._last_boosts_available = self.boosts_available

        return reward

    def _check_termination(self):
        if self.collision_count >= self.MAX_COLLISIONS:
            self.score -= 100
            return True
        if self.time_elapsed >= self.TIME_LIMIT_SECONDS:
            self.score -= 100
            return True
        if self.current_checkpoint_index >= len(self.checkpoints):
            time_bonus = max(0, self.TIME_LIMIT_SECONDS - self.time_elapsed)
            self.score += 100 + (time_bonus * 2)
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        if self.player_pos is None: return

        offset = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2) - self.player_pos
        if self.camera_shake > 0:
            offset += pygame.math.Vector2(self.np_random.integers(-5, 6), self.np_random.integers(-5, 6))

        for i in range(len(self.track_centerline) - 1):
            p1 = self.track_centerline[i] + offset
            p2 = self.track_centerline[i+1] + offset
            pygame.draw.line(self.screen, self.COLOR_TRACK, p1, p2, self.TRACK_WIDTH)
            # FIX: The original call had 6 arguments, but pygame.draw.line takes at most 5.
            # The extra '1' is removed. This results in this line overdrawing the one above,
            # effectively changing the track color to COLOR_TRACK_BORDER. This is the most
            # direct fix for the TypeError.
            pygame.draw.line(self.screen, self.COLOR_TRACK_BORDER, p1, p2, self.TRACK_WIDTH)
            pygame.gfxdraw.line(self.screen, int(p1.x), int(p1.y), int(p2.x), int(p2.y), self.COLOR_TRACK_BORDER)

        finish_line_start = self.track_centerline[-2]
        finish_line_end = self.track_centerline[-1]
        line_vec = (finish_line_end - finish_line_start).normalize()
        perp_vec = pygame.math.Vector2(-line_vec.y, line_vec.x)
        
        for i in range(-6, 7, 2):
            for j in range(-int(self.TRACK_WIDTH/20), int(self.TRACK_WIDTH/20)):
                color = self.COLOR_FINISH_LIGHT if (i+j) % 2 == 0 else self.COLOR_FINISH_DARK
                start_pos = finish_line_end + perp_vec * j * 10 + line_vec * i * 2 + offset
                pygame.draw.rect(self.screen, color, (start_pos.x, start_pos.y, 10, 10))

        for i, cp in enumerate(self.checkpoints):
            if i >= self.current_checkpoint_index:
                pos = cp + offset
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(self.TRACK_WIDTH/2), (*self.COLOR_CHECKPOINT, 50))
                pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(self.TRACK_WIDTH/2), self.COLOR_CHECKPOINT)

        for obs in self.obstacles:
            pos = obs['pos'] + offset
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (pos.x - obs['radius'], pos.y - obs['radius'], obs['radius']*2, obs['radius']*2))

        for boost in self.boosts:
            pos = boost['pos'] + offset
            points = [(pos.x, pos.y - 10), (pos.x - 8.66, pos.y + 5), (pos.x + 8.66, pos.y + 5)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_BOOST)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_BOOST)

        for p in self.particles:
            pos = p['pos'] + offset
            size = max(1, p['size'] * (p['life'] / p['max_life']))
            pygame.draw.circle(self.screen, p['color'], pos, size)

        player_screen_pos = self.player_pos + offset
        
        glow_surf = pygame.Surface((60, 60), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (30, 30), 30)
        self.screen.blit(glow_surf, (player_screen_pos.x - 30, player_screen_pos.y - 30))

        car_points = [pygame.math.Vector2(15, 0), pygame.math.Vector2(-10, -8), pygame.math.Vector2(-10, 8)]
        rotated_points = [p.rotate(-math.degrees(self.player_angle)) + player_screen_pos for p in car_points]
        
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PLAYER)

    def _render_ui(self):
        time_text = f"TIME: {self.TIME_LIMIT_SECONDS - self.time_elapsed:.1f}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (10, 10))
        
        collision_text = f"HITS: {self.collision_count}/{self.MAX_COLLISIONS}"
        collision_surf = self.font_small.render(collision_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(collision_surf, (self.SCREEN_WIDTH - collision_surf.get_width() - 10, 10))

        boost_text = f"BOOSTS: {self.boosts_available}"
        boost_surf = self.font_small.render(boost_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(boost_surf, (self.SCREEN_WIDTH - boost_surf.get_width() - 10, 35))

        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 35))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "GAME OVER"
            if self.current_checkpoint_index >= len(self.checkpoints): msg = "FINISH!"
            elif self.collision_count >= self.MAX_COLLISIONS: msg = "WRECKED!"
            elif self.time_elapsed >= self.TIME_LIMIT_SECONDS: msg = "TIME UP!"

            end_text_surf = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            self.screen.blit(end_text_surf, (self.SCREEN_WIDTH/2 - end_text_surf.get_width()/2, self.SCREEN_HEIGHT/2 - end_text_surf.get_height()/2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed,
            "collision_count": self.collision_count,
            "checkpoints_cleared": self.current_checkpoint_index
        }

    def _generate_track(self):
        self.track_centerline = []
        num_segments = 25
        segment_length = 250
        amplitude = 150
        frequency = 0.5
        
        start_pos = pygame.math.Vector2(200, self.SCREEN_HEIGHT / 2)
        self.player_pos = start_pos.copy()

        for i in range(num_segments + 2):
            x = start_pos.x + i * segment_length
            y = start_pos.y + math.sin(i * frequency) * self.np_random.uniform(0.8, 1.2)
            self.track_centerline.append(pygame.math.Vector2(x, y))

        self.checkpoints = self.track_centerline[4::5]
        self.current_checkpoint_index = 0

    def _populate_track(self):
        self.obstacles = []
        self.boosts = []
        self._last_collision_count = 0
        self._last_boosts_available = 0

        for i in range(len(self.track_centerline) - 1):
            for _ in range(2):
                self._spawn_obstacle(segment_index=i)
            
            if i > 0 and i % 4 == 0:
                p1 = self.track_centerline[i]
                p2 = self.track_centerline[i+1]
                lerp_factor = self.np_random.uniform(0.2, 0.8)
                pos = p1.lerp(p2, lerp_factor)
                self.boosts.append({'pos': pos})

    def _spawn_obstacle(self, segment_index=None):
        if segment_index is None:
            _, _, p_idx = self._get_closest_track_segment()
            if p_idx is None: return
            segment_index = min(len(self.track_centerline)-2, p_idx + self.np_random.integers(2, 5))

        p1 = self.track_centerline[segment_index]
        p2 = self.track_centerline[segment_index+1]
        
        lerp_factor = self.np_random.uniform(0.1, 0.9)
        pos_on_centerline = p1.lerp(p2, lerp_factor)
        
        track_dir = (p2 - p1).normalize()
        perp_dir = pygame.math.Vector2(-track_dir.y, track_dir.x)
        
        offset_dist = self.np_random.uniform(20, self.TRACK_WIDTH / 2 - 20)
        offset_dir = self.np_random.choice([-1, 1])
        
        pos = pos_on_centerline + perp_dir * offset_dist * offset_dir
        
        if all(pos.distance_to(o['pos']) > 50 for o in self.obstacles):
            self.obstacles.append({'pos': pos, 'radius': self.np_random.uniform(8, 12)})

    def _get_dist_to_next_checkpoint(self):
        if self.player_pos is None: return float('inf')
        if self.current_checkpoint_index >= len(self.checkpoints):
            return self.player_pos.distance_to(self.track_centerline[-1])
        return self.player_pos.distance_to(self.checkpoints[self.current_checkpoint_index])

    def _get_closest_track_segment(self):
        if not self.track_centerline or self.player_pos is None:
            return None, None, None
            
        min_dist_sq = float('inf')
        closest_p1, closest_p2 = None, None
        closest_idx = -1

        start_idx = max(0, self.current_checkpoint_index * 5 - 10)
        end_idx = min(len(self.track_centerline) - 1, self.current_checkpoint_index * 5 + 10)
        if end_idx <= start_idx:
            start_idx = 0
            end_idx = len(self.track_centerline) - 1

        for i in range(start_idx, end_idx):
            p1 = self.track_centerline[i]
            p2 = self.track_centerline[i+1]
            dist_sq = self.player_pos.distance_squared_to(p1.lerp(p2, 0.5))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_p1, closest_p2 = p1, p2
                closest_idx = i
        return closest_p1, closest_p2, closest_idx

    def _create_particles(self, pos, count, color, life, speed, angle_offset=0, angle_spread=2*math.pi):
        for _ in range(count):
            angle = self.np_random.uniform(angle_offset - angle_spread/2, angle_offset + angle_spread/2)
            vel = pygame.math.Vector2()
            vel.from_polar((self.np_random.uniform(0.5, 1) * speed, -math.degrees(angle)))
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'color': color,
                'life': self.np_random.uniform(0.5, 1) * life,
                'max_life': life, 'size': self.np_random.uniform(2, 5)
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    
    running = True
    terminated, truncated = False, False
    
    env.auto_advance = False
    display = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Arcade Racer")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                terminated, truncated = False, False
                obs, info = env.reset(seed=42)

        if terminated or truncated:
            pygame.time.wait(100)
            continue

        keys = pygame.key.get_pressed()
        
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        
        if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
             movement = 3
        elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
             movement = 4
        
        if keys[pygame.K_UP] and keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_UP] and keys[pygame.K_RIGHT]: movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Time: {info['time_elapsed']:.2f}s. Press 'R' to restart.")
        
        env.clock.tick(env.FPS)
        
    env.close()