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

    user_guide = (
        "Controls: ↑ to accelerate, ↓ to brake/reverse, ←→ to turn. "
        "Hold Shift to drift. Press Space for a speed boost."
    )

    game_description = (
        "A fast-paced arcade racer. Weave through a procedurally generated track, "
        "dodge obstacles, and race against the clock to reach the finish line."
    )

    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_TRACK = (50, 50, 65)
    COLOR_LINES = (200, 200, 220)
    COLOR_PLAYER = (255, 60, 60)
    COLOR_PLAYER_GLOW = (255, 100, 100, 64)
    COLOR_OBSTACLE = (60, 120, 255)
    COLOR_OBSTACLE_GLOW = (100, 150, 255, 64)
    COLOR_CHECKPOINT = (60, 255, 120, 128)
    COLOR_PARTICLE = (255, 220, 100)
    COLOR_TEXT = (255, 255, 255)
    COLOR_SHADOW = (0, 0, 0, 100)

    # Player Physics
    ACCELERATION = 0.2
    BRAKE_FORCE = 0.4
    MAX_SPEED = 8.0
    MAX_REVERSE_SPEED = -2.0
    FRICTION = 0.97
    TURN_SPEED = 0.05
    
    # Drift Physics
    DRIFT_TURN_MOD = 1.5
    DRIFT_FRICTION = 0.98
    DRIFT_SLIDE_FACTOR = 0.8
    
    # Boost
    BOOST_ACCELERATION = 0.8
    BOOST_DURATION = 30 # steps
    BOOST_COOLDOWN = 150 # steps

    # Game Rules
    MAX_COLLISIONS = 5
    GAME_TIME_LIMIT = 60.0  # seconds
    FPS = 30
    MAX_STEPS = int(GAME_TIME_LIMIT * FPS)
    
    # Track Generation
    TRACK_WIDTH = 150
    TRACK_LENGTH = 15000
    SEGMENT_LENGTH = 150
    CURVE_FACTOR = 0.8
    CHECKPOINT_INTERVAL = 15 # segments
    OBSTACLE_DENSITY_START = 0.1
    OBSTACLE_DENSITY_INCREASE = 0.05 # per 10 seconds

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
        
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 36)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 20)
        except pygame.error:
            self.font_large = pygame.font.SysFont("sans-serif", 36)
            self.font_small = pygame.font.SysFont("sans-serif", 20)

        self.render_mode = render_mode
        self.particles = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self._generate_track()
        
        # FIX: Start player at the beginning of the track (high Y), not the end.
        self.player_pos = self.track_centerline[0].copy()
        self.player_angle = -math.pi / 2  # Start facing up the track
        self.player_speed = 0.0
        self.player_velocity = np.array([0.0, 0.0])
        
        self.collision_count = 0
        self.game_timer = self.GAME_TIME_LIMIT
        
        self.boost_timer = 0
        self.boost_cooldown = 0
        
        self.particles.clear()

        self.last_dist_to_finish = self._get_dist_to_finish()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._update_player(movement, space_held, shift_held)
        self._update_game_state()
        
        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True # Ensure game state is consistent
        
        self.steps += 1
        if self.auto_advance:
            self.clock.tick(self.FPS)

        obs = self._get_observation()
        return obs, reward, terminated, truncated, self._get_info()

    def _generate_track(self):
        self.track_centerline = []
        current_pos = np.array([self.SCREEN_WIDTH / 2.0, float(self.TRACK_LENGTH - 200)])
        angle = -math.pi / 2
        
        num_segments = int(self.TRACK_LENGTH / self.SEGMENT_LENGTH)
        
        for i in range(num_segments):
            self.track_centerline.append(current_pos.copy())
            angle += self.np_random.uniform(-1, 1) * self.CURVE_FACTOR * (i / num_segments)
            angle = np.clip(angle, -math.pi/2 - 0.8, -math.pi/2 + 0.8)
            
            move_vec = np.array([math.cos(angle), math.sin(angle)])
            current_pos += move_vec * self.SEGMENT_LENGTH

        # FIX: Finish line should be at the end of the track (low Y).
        self.finish_line_y = self.track_centerline[-1][1]
        
        self._generate_checkpoints()
        self._generate_obstacles()

    def _generate_checkpoints(self):
        self.checkpoints = []
        for i in range(self.CHECKPOINT_INTERVAL, len(self.track_centerline), self.CHECKPOINT_INTERVAL):
            pos = self.track_centerline[i]
            self.checkpoints.append({"pos": pos, "passed": False})

    def _generate_obstacles(self):
        self.obstacles = []
        for i in range(len(self.track_centerline) - 1):
            p1 = self.track_centerline[i]
            p2 = self.track_centerline[i+1]
            
            time_progress = (self.TRACK_LENGTH - p1[1]) / self.TRACK_LENGTH
            current_density = self.OBSTACLE_DENSITY_START + self.OBSTACLE_DENSITY_INCREASE * (time_progress * self.GAME_TIME_LIMIT / 10)

            if self.np_random.random() < current_density:
                t = self.np_random.random()
                center_pos = p1 * (1-t) + p2 * t
                
                direction = p2 - p1
                norm_direction = np.linalg.norm(direction)
                if norm_direction == 0: continue
                perp_vec = np.array([-direction[1], direction[0]]) / norm_direction
                
                offset = self.np_random.uniform(-self.TRACK_WIDTH * 0.4, self.TRACK_WIDTH * 0.4)
                
                if abs(offset) < 20:
                    offset = np.sign(offset) * 20 if offset != 0 else 20

                obstacle_pos = center_pos + perp_vec * offset
                radius = self.np_random.uniform(8, 15)
                self.obstacles.append({'pos': obstacle_pos, 'radius': radius})

    def _update_player(self, movement, space_held, shift_held):
        if self.boost_cooldown > 0:
            self.boost_cooldown -= 1
        if self.boost_timer > 0:
            self.boost_timer -= 1
        
        if space_held and self.boost_cooldown == 0 and self.boost_timer == 0:
            self.boost_timer = self.BOOST_DURATION
            self.boost_cooldown = self.BOOST_COOLDOWN
            self._create_particles(self.player_pos, 20, speed_min=2, speed_max=5, color=(255,150,50))

        if self.boost_timer > 0:
            self.player_speed += self.BOOST_ACCELERATION
            
        turn_mod = self.DRIFT_TURN_MOD if shift_held else 1.0
        if movement == 3:  # Left
            self.player_angle -= self.TURN_SPEED * turn_mod * (1 - self.player_speed / (self.MAX_SPEED * 2))
        if movement == 4:  # Right
            self.player_angle += self.TURN_SPEED * turn_mod * (1 - self.player_speed / (self.MAX_SPEED * 2))

        if movement == 1:  # Up
            self.player_speed += self.ACCELERATION
        elif movement == 2:  # Down
            self.player_speed -= self.BRAKE_FORCE
        
        friction = self.DRIFT_FRICTION if shift_held else self.FRICTION
        self.player_speed *= friction
        self.player_speed = np.clip(self.player_speed, self.MAX_REVERSE_SPEED, self.MAX_SPEED)

        forward_vec = np.array([math.cos(self.player_angle), math.sin(self.player_angle)])
        
        if shift_held:
            self.player_velocity = self.player_velocity * self.DRIFT_SLIDE_FACTOR + forward_vec * self.player_speed * (1-self.DRIFT_SLIDE_FACTOR)
            speed = np.linalg.norm(self.player_velocity)
            if speed > self.MAX_SPEED:
                self.player_velocity = (self.player_velocity / speed) * self.MAX_SPEED
            self.player_speed = speed
        else:
            self.player_velocity = forward_vec * self.player_speed

        self.player_pos += self.player_velocity

    def _update_game_state(self):
        self.game_timer -= 1 / self.FPS
        
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['vel'] *= 0.95

    def _calculate_reward(self):
        reward = 0
        reward -= 0.01

        dist_to_finish = self._get_dist_to_finish()
        reward += (self.last_dist_to_finish - dist_to_finish) * 0.2
        self.last_dist_to_finish = dist_to_finish
        
        for cp in self.checkpoints:
            if not cp['passed'] and np.linalg.norm(self.player_pos - cp['pos']) < self.TRACK_WIDTH / 2:
                cp['passed'] = True
                reward += 5
                self.score += 5
                self._create_particles(self.player_pos, 30, color=self.COLOR_CHECKPOINT, life=40, speed_max=4)

        collided = self._handle_collisions()
        if collided:
            reward -= 10
        
        if self.victory:
            reward += 100
        elif self.game_over and not self.victory:
            reward -= 50

        return reward

    def _handle_collisions(self):
        collided_this_step = False
        player_radius = 10

        for obs in self.obstacles:
            dist = np.linalg.norm(self.player_pos - obs['pos'])
            if dist < obs['radius'] + player_radius:
                self.collision_count += 1
                collided_this_step = True
                self.player_speed *= 0.5
                
                repel_vec_norm = np.linalg.norm(self.player_pos - obs['pos'])
                repel_vec = (self.player_pos - obs['pos']) / (repel_vec_norm if repel_vec_norm != 0 else 1)
                self.player_pos += repel_vec * (obs['radius'] + player_radius - dist)
                self.player_velocity = repel_vec * 2

                self._create_particles(self.player_pos, 50, color=self.COLOR_PARTICLE)
                break

        min_dist = float('inf')
        closest_segment_idx = -1
        for i in range(len(self.track_centerline)):
            dist = np.linalg.norm(self.player_pos - self.track_centerline[i])
            if dist < min_dist:
                min_dist = dist
                closest_segment_idx = i
        
        if closest_segment_idx != -1 and closest_segment_idx + 1 < len(self.track_centerline):
            p1 = self.track_centerline[closest_segment_idx]
            p2 = self.track_centerline[closest_segment_idx + 1]

            l2 = np.linalg.norm(p2 - p1)**2
            if l2 > 0:
                t = max(0, min(1, np.dot(self.player_pos - p1, p2 - p1) / l2))
                closest_point_on_line = p1 + t * (p2 - p1)
                dist_to_centerline = np.linalg.norm(self.player_pos - closest_point_on_line)
            
                if dist_to_centerline > self.TRACK_WIDTH / 2:
                    self.player_speed *= 0.9
                    direction_to_center = closest_point_on_line - self.player_pos
                    self.player_pos += direction_to_center * (1 - (self.TRACK_WIDTH / 2) / dist_to_centerline)

        return collided_this_step

    def _check_termination(self):
        if self.collision_count >= self.MAX_COLLISIONS:
            self.game_over = True
        if self.game_timer <= 0:
            self.game_over = True
        if self.player_pos[1] < self.finish_line_y:
            self.game_over = True
            self.victory = True
        
        return self.game_over

    def _get_dist_to_finish(self):
        return self.player_pos[1]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.game_timer,
            "collisions": self.collision_count,
            "victory": self.victory,
        }

    def _world_to_screen(self, world_pos):
        screen_center = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])
        return (world_pos - self.player_pos + screen_center).astype(int)

    def _render_game(self):
        self._render_track()
        self._render_checkpoints()
        self._render_obstacles()
        self._render_particles()
        self._render_player()

    def _render_track(self):
        for i in range(len(self.track_centerline) - 1):
            p1 = self._world_to_screen(self.track_centerline[i])
            p2 = self._world_to_screen(self.track_centerline[i+1])
            
            if max(p1[1], p2[1]) < 0 or min(p1[1], p2[1]) > self.SCREEN_HEIGHT:
                continue
            
            pygame.draw.line(self.screen, self.COLOR_TRACK, p1, p2, width=self.TRACK_WIDTH)
        
        for i in range(len(self.track_centerline) - 1):
            p1 = self.track_centerline[i]
            p2 = self.track_centerline[i+1]
            
            direction = p2 - p1
            norm_direction = np.linalg.norm(direction)
            if norm_direction == 0: continue
            perp_vec = np.array([-direction[1], direction[0]]) / norm_direction
            
            l1 = self._world_to_screen(p1 + perp_vec * self.TRACK_WIDTH / 2)
            r1 = self._world_to_screen(p1 - perp_vec * self.TRACK_WIDTH / 2)
            l2 = self._world_to_screen(p2 + perp_vec * self.TRACK_WIDTH / 2)
            r2 = self._world_to_screen(p2 - perp_vec * self.TRACK_WIDTH / 2)

            pygame.draw.aaline(self.screen, self.COLOR_LINES, l1, l2)
            pygame.draw.aaline(self.screen, self.COLOR_LINES, r1, r2)
            
        finish_y_screen = self._world_to_screen(np.array([0, self.finish_line_y]))[1]
        if 0 < finish_y_screen < self.SCREEN_HEIGHT:
            for i in range(0, self.SCREEN_WIDTH, 20):
                color = self.COLOR_LINES if (i // 20) % 2 == 0 else self.COLOR_BG
                pygame.draw.rect(self.screen, color, (i, finish_y_screen, 20, 10))

    def _render_checkpoints(self):
        for cp in self.checkpoints:
            if not cp['passed']:
                pos = self._world_to_screen(cp['pos'])
                if -self.TRACK_WIDTH < pos[0] < self.SCREEN_WIDTH + self.TRACK_WIDTH and \
                   -self.TRACK_WIDTH < pos[1] < self.SCREEN_HEIGHT + self.TRACK_WIDTH:
                    s = pygame.Surface((self.TRACK_WIDTH, 20), pygame.SRCALPHA)
                    s.fill(self.COLOR_CHECKPOINT)
                    
                    min_dist = float('inf')
                    closest_idx = -1
                    for i in range(len(self.track_centerline)):
                        dist = np.linalg.norm(cp['pos'] - self.track_centerline[i])
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = i
                    
                    if closest_idx > 0:
                        p1 = self.track_centerline[closest_idx-1]
                        p2 = self.track_centerline[closest_idx]
                        angle_rad = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
                        angle_deg = -math.degrees(angle_rad)
                        
                        rotated_s = pygame.transform.rotate(s, angle_deg)
                        rect = rotated_s.get_rect(center=pos)
                        self.screen.blit(rotated_s, rect)

    def _render_obstacles(self):
        for obs in self.obstacles:
            pos = self._world_to_screen(obs['pos'])
            radius = int(obs['radius'])
            if pos[0] + radius > 0 and pos[0] - radius < self.SCREEN_WIDTH and \
               pos[1] + radius > 0 and pos[1] - radius < self.SCREEN_HEIGHT:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_OBSTACLE)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_OBSTACLE)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 3, self.COLOR_OBSTACLE_GLOW)


    def _render_player(self):
        car_w, car_h = 12, 24
        screen_center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)

        car_surf = pygame.Surface((car_h, car_w), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, self.COLOR_PLAYER, (0, 0, car_h, car_w), border_radius=3)
        
        pygame.draw.rect(car_surf, (255, 255, 200), (car_h - 4, 1, 4, 3))
        pygame.draw.rect(car_surf, (255, 255, 200), (car_h - 4, car_w - 4, 4, 3))

        angle_deg = -math.degrees(self.player_angle) - 90
        rotated_car = pygame.transform.rotate(car_surf, angle_deg)
        car_rect = rotated_car.get_rect(center=screen_center)

        if self.boost_timer > 0:
            glow_surf = pygame.Surface((car_h*2, car_h*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (255,180,50, 100), (car_h, car_h), car_h)
            pygame.draw.circle(glow_surf, (255,180,50, 50), (car_h, car_h), car_h*0.7)
            glow_rect = glow_surf.get_rect(center=screen_center)
            self.screen.blit(glow_surf, glow_rect)

        self.screen.blit(rotated_car, car_rect)

    def _render_particles(self):
        for p in self.particles:
            pos = self._world_to_screen(p['pos'])
            pygame.draw.circle(self.screen, p['color'], pos, int(p['life'] * p['size_mod']), 1)

    def _render_ui(self):
        time_text = f"TIME: {max(0, self.game_timer):.2f}"
        self._draw_text(time_text, self.font_small, (10, 10))
        
        collision_text = f"HITS: {self.collision_count}/{self.MAX_COLLISIONS}"
        text_surf = self.font_small.render(collision_text, True, self.COLOR_TEXT)
        self._draw_text(collision_text, self.font_small, (self.SCREEN_WIDTH - text_surf.get_width() - 10, 10))

        boost_ready = self.boost_cooldown == 0
        boost_color = (100, 255, 100) if boost_ready else (200, 100, 100)
        
        bar_w = 100
        pygame.draw.rect(self.screen, (50,50,50), (10, 40, bar_w, 15))
        if boost_ready:
            fill_w = bar_w
        else:
            fill_w = int(bar_w * (1 - self.boost_cooldown / self.BOOST_COOLDOWN))
        pygame.draw.rect(self.screen, boost_color, (10, 40, fill_w, 15))
        self._draw_text("BOOST", self.font_small, (15 + bar_w, 38), size=16)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY!" if self.victory else "GAME OVER"
            self._draw_text(message, self.font_large, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20), center=True)

    def _draw_text(self, text, font, pos, color=COLOR_TEXT, center=False, size=None):
        if size:
             font = pygame.font.Font(pygame.font.get_default_font(), size)

        text_surface = font.render(text, True, color)
        text_shadow = font.render(text, True, (0,0,0))
        
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(text_shadow, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surface, text_rect)

    def _create_particles(self, pos, count, color=(255, 200, 0), life=20, speed_min=1, speed_max=3):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(speed_min, speed_max)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(life // 2, life),
                'color': color,
                'size_mod': self.np_random.uniform(0.1, 0.2)
            })

    def close(self):
        pygame.quit()