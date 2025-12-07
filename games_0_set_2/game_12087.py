import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:18:15.298073
# Source Brief: brief_02087.md
# Brief Index: 2087
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Neon Tunnel Racer
    Race through a procedurally generated, visually stunning geometric maze
    against the clock, chaining perfect turns for maximum points.
    """
    metadata = {"render_modes": ["rgb_array", "human_playable"]}

    game_description = (
        "Race through a procedurally generated neon tunnel against the clock. "
        "Chain perfect turns and use boosts to achieve the highest score."
    )
    user_guide = (
        "Controls: Use ↑↓←→ or WASD to move your ship. Press space to activate a speed boost."
    )
    auto_advance = True

    # Class attribute for difficulty progression across episodes
    difficulty_level = 1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.render_mode = render_mode
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 40, bold=True)

        # --- Game Constants ---
        self.MAX_STEPS = 2000
        self.MAX_TIME = 60.0  # seconds
        self.INITIAL_LIVES = 3
        self.PLAYER_SPEED = 1.5
        self.PLAYER_FRICTION = 0.92
        self.BOOST_SPEED_MULTIPLIER = 3.0
        self.BOOST_DURATION = 15  # steps
        self.BOOST_COOLDOWN = 45  # steps
        self.FOCAL_LENGTH = 250
        self.DRAW_DISTANCE = 40

        # --- Colors ---
        self.COLOR_BG = (10, 0, 25)
        self.COLOR_WALL = (30, 20, 80)
        self.COLOR_WALL_TURN = (120, 40, 100)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 150)
        self.COLOR_BOOST_TRAIL = (0, 255, 100)
        self.COLOR_UI_TEXT = (220, 220, 255)
        self.COLOR_TIME_BAR = (0, 180, 255)
        self.COLOR_TIME_BAR_WARN = (255, 100, 0)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.path = []
        self.path_length = 0
        self.path_progress = 0.0
        self.time_remaining = 0.0
        self.lives = 0
        self.combo = 0
        self.boost_timer = 0
        self.boost_cooldown_timer = 0
        self.particles = []
        self.screen_shake = 0
        self.last_segment_idx = -1
        self.won_last_game = False

        # self.reset() is called by the environment wrapper, no need to call it here.
        
    def _generate_path(self):
        path = []
        num_segments = 150
        segment_length = 20
        
        # Difficulty scaling
        current_width = 150 * (0.95 ** (GameEnv.difficulty_level - 1))
        turn_chance = 0.15 * (1.10 ** (GameEnv.difficulty_level - 1))
        turn_chance = min(turn_chance, 0.6)

        current_angle = 0
        current_offset_x = 0
        
        for i in range(num_segments):
            is_turn_segment = False
            # Ensure path is initially straight and has a straight finish
            if i > 5 and i < num_segments - 10 and random.random() < turn_chance:
                is_turn_segment = True
                turn_direction = random.choice([-1, 1])
                turn_angle_increment = random.uniform(0.03, 0.05) * turn_direction
                turn_length = random.randint(15, 25)
                for _ in range(turn_length):
                    current_angle += turn_angle_increment
                    current_offset_x += math.sin(current_angle) * segment_length
                    path.append({
                        'z': len(path) * segment_length,
                        'offset': pygame.Vector2(current_offset_x, 0),
                        'width': current_width,
                        'is_turn': True
                    })
                continue

            # Straight segment
            current_offset_x += math.sin(current_angle) * segment_length
            path.append({
                'z': len(path) * segment_length,
                'offset': pygame.Vector2(current_offset_x, 0),
                'width': current_width,
                'is_turn': is_turn_segment
            })
        
        self.path_length = len(path) * segment_length
        return path

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.won_last_game:
            GameEnv.difficulty_level += 1
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.path_progress = 0.0
        self.time_remaining = self.MAX_TIME
        self.lives = self.INITIAL_LIVES
        self.combo = 1
        self.boost_timer = 0
        self.boost_cooldown_timer = 0
        self.particles = []
        self.screen_shake = 0
        self.won_last_game = False
        
        self.player_pos.update(0, 0)
        self.player_vel.update(0, 0)
        
        self.path = self._generate_path()
        self.last_segment_idx = -1
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.0
        
        # --- Update Timers ---
        self.steps += 1
        self.time_remaining -= 1.0 / 30.0 # Assuming 30 FPS
        if self.boost_timer > 0: self.boost_timer -= 1
        if self.boost_cooldown_timer > 0: self.boost_cooldown_timer -= 1
        if self.screen_shake > 0: self.screen_shake -= 1

        # --- Player Input & Movement ---
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -self.PLAYER_SPEED
        elif movement == 2: move_vec.y = self.PLAYER_SPEED
        elif movement == 3: move_vec.x = -self.PLAYER_SPEED
        elif movement == 4: move_vec.x = self.PLAYER_SPEED
        
        self.player_vel += move_vec
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_FRICTION

        # --- Boost Logic ---
        is_boosting = self.boost_timer > 0
        boost_activated_this_step = False
        if space_held and self.boost_cooldown_timer == 0 and self.boost_timer == 0:
            # sfx: boost_activate
            self.boost_timer = self.BOOST_DURATION
            self.boost_cooldown_timer = self.BOOST_COOLDOWN
            is_boosting = True
            boost_activated_this_step = True

        # --- Forward Progress ---
        forward_speed = 10.0
        if is_boosting:
            forward_speed *= self.BOOST_SPEED_MULTIPLIER
        self.path_progress += forward_speed
        
        # --- Particle Management ---
        trail_color = self.COLOR_BOOST_TRAIL if is_boosting else self.COLOR_PLAYER_GLOW
        for _ in range(3 if is_boosting else 1):
            p_offset = pygame.Vector2(random.uniform(-10, 10), random.uniform(-10, 10))
            self.particles.append({
                'pos': self.player_pos + p_offset,
                'vel': self.player_vel * 0.5 + pygame.Vector2(random.uniform(-1,1), random.uniform(-1,1)),
                'life': 20, 'color': trail_color, 'radius': random.uniform(2, 5)
            })
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # --- Collision Detection and Reward Calculation ---
        current_segment_idx = int(self.path_progress / 20)
        if 0 <= current_segment_idx < len(self.path):
            segment = self.path[current_segment_idx]
            tunnel_center_x = segment['offset'].x
            tunnel_radius = segment['width']
            
            player_dist_from_center = abs(self.player_pos.x - tunnel_center_x)
            
            if player_dist_from_center > tunnel_radius or abs(self.player_pos.y) > tunnel_radius:
                # sfx: collision_sound
                self.lives -= 1
                reward -= 50.0
                self.combo = 1
                self.screen_shake = 10
                if player_dist_from_center > tunnel_radius:
                  self.player_vel.x *= -0.8
                  self.player_pos.x = tunnel_center_x + tunnel_radius * np.sign(self.player_pos.x - tunnel_center_x)
                if abs(self.player_pos.y) > tunnel_radius:
                  self.player_vel.y *= -0.8
                  self.player_pos.y = tunnel_radius * np.sign(self.player_pos.y)
            else:
                reward += 0.1 # Survival reward
                if player_dist_from_center < tunnel_radius * 0.2:
                    reward += 0.5 # Center reward
            
            if segment['is_turn'] and self.last_segment_idx != current_segment_idx:
                reward += 1.0 * self.combo
                self.combo = min(self.combo + 1, 10)
                if boost_activated_this_step:
                    # sfx: perfect_boost
                    reward += 5.0
            self.last_segment_idx = current_segment_idx
        else:
            self.path_progress = self.path_length

        self.score += reward

        # --- Termination Conditions ---
        terminated = False
        if self.lives <= 0:
            # sfx: game_over_crash
            reward -= 50.0 # Final penalty
            terminated = True
        if self.time_remaining <= 0:
            # sfx: game_over_timeout
            reward -= 100.0
            terminated = True
        if self.path_progress >= self.path_length:
            # sfx: victory
            reward += 100.0
            terminated = True
            self.won_last_game = True
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        if terminated:
            self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _project(self, point_3d, camera_offset):
        z_dist = point_3d.z - self.path_progress
        if z_dist <= 0: return None
        
        scale = self.FOCAL_LENGTH / z_dist
        x = self.width / 2 + (point_3d.x - camera_offset.x) * scale
        y = self.height / 2 + (point_3d.y - camera_offset.y) * scale
        
        return pygame.Vector2(x, y)

    def _render_tunnel(self):
        start_idx = max(0, int(self.path_progress / 20) - 2)
        end_idx = min(len(self.path), start_idx + self.DRAW_DISTANCE)
        
        cam_idx = int(self.path_progress / 20)
        camera_offset = pygame.Vector2(0, 0)
        if cam_idx + 1 < len(self.path):
            p1 = self.path[cam_idx]
            p2 = self.path[cam_idx + 1]
            interp = (self.path_progress - p1['z']) / 20.0
            camera_offset = p1['offset'].lerp(p2['offset'], interp)
        elif cam_idx < len(self.path):
            camera_offset = self.path[cam_idx]['offset']
        else:
            # Reached end of path, find last valid offset
            if len(self.path) > 0:
                camera_offset = self.path[-1]['offset']

        for i in range(start_idx, end_idx - 1):
            p1_data, p2_data = self.path[i], self.path[i+1]
            color = self.COLOR_WALL_TURN if p1_data['is_turn'] else self.COLOR_WALL
            
            num_vertices = 8
            for j in range(num_vertices):
                angle1, angle2 = (j/num_vertices)*2*math.pi, ((j+1)/num_vertices)*2*math.pi
                
                v1_p1 = pygame.Vector3(p1_data['offset'].x + math.cos(angle1)*p1_data['width'], math.sin(angle1)*p1_data['width'], p1_data['z'])
                v2_p1 = pygame.Vector3(p1_data['offset'].x + math.cos(angle2)*p1_data['width'], math.sin(angle2)*p1_data['width'], p1_data['z'])
                v1_p2 = pygame.Vector3(p2_data['offset'].x + math.cos(angle1)*p2_data['width'], math.sin(angle1)*p2_data['width'], p2_data['z'])
                v2_p2 = pygame.Vector3(p2_data['offset'].x + math.cos(angle2)*p2_data['width'], math.sin(angle2)*p2_data['width'], p2_data['z'])
                
                proj_v1_p1, proj_v2_p1 = self._project(v1_p1, camera_offset), self._project(v2_p1, camera_offset)
                proj_v1_p2, proj_v2_p2 = self._project(v1_p2, camera_offset), self._project(v2_p2, camera_offset)

                if all([proj_v1_p1, proj_v2_p1, proj_v1_p2, proj_v2_p2]):
                    points = [proj_v1_p1, proj_v2_p1, proj_v2_p2, proj_v1_p2]
                    try:
                        pygame.gfxdraw.aapolygon(self.screen, points, color)
                        pygame.gfxdraw.filled_polygon(self.screen, points, color)
                    except (ValueError, TypeError): pass # Ignore degenerate polygons
                        
    def _render_player_and_particles(self):
        for p in sorted(self.particles, key=lambda x: x['life']):
            alpha = p['life'] / 20.0
            color = (*p['color'], int(alpha * 255))
            radius = int(p['radius'] * alpha)
            if radius > 0:
                particle_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (radius, radius), radius)
                pos = (self.width/2 + p['pos'].x - radius, self.height/2 + p['pos'].y - radius)
                self.screen.blit(particle_surf, pos, special_flags=pygame.BLEND_RGBA_ADD)

        player_screen_pos = (int(self.width/2 + self.player_pos.x), int(self.height/2 + self.player_pos.y))
        
        glow_radius = 25
        glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 80), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (player_screen_pos[0]-glow_radius, player_screen_pos[1]-glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, player_screen_pos, 10)

    def _render_ui(self):
        time_ratio = max(0, self.time_remaining / self.MAX_TIME)
        bar_color = self.COLOR_TIME_BAR if time_ratio > 0.25 else self.COLOR_TIME_BAR_WARN
        pygame.draw.rect(self.screen, bar_color, (0, 0, self.width * time_ratio, 10))

        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 20))
        
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.width - lives_text.get_width() - 10, 20))

        diff_text = self.font_small.render(f"LEVEL: {self.difficulty_level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(diff_text, (self.width - diff_text.get_width() - 10, 40))

        if self.combo > 1:
            combo_text = self.font_large.render(f"x{self.combo}", True, self.COLOR_UI_TEXT)
            self.screen.blit(combo_text, combo_text.get_rect(center=(self.width/2, self.height - 50)))
            
        cooldown_ratio = self.boost_cooldown_timer / self.BOOST_COOLDOWN if self.BOOST_COOLDOWN > 0 else 0
        if cooldown_ratio > 0:
            pygame.draw.rect(self.screen, (50, 50, 80), (self.width/2 - 52, self.height - 22, 104, 14))
            pygame.draw.rect(self.screen, (100, 100, 200), (self.width/2 - 50, self.height - 20, 100 * (1-cooldown_ratio), 10))
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        render_target = self.screen
        if self.screen_shake > 0:
            shake_offset = (random.randint(-8, 8), random.randint(-8, 8))
            # Create a temporary surface for the shake effect to avoid modifying the main screen buffer directly
            # before all rendering is complete.
            temp_surface = pygame.Surface(self.screen.get_size())
            temp_surface.blit(self.screen, (0,0)) # Copy current state
            # This logic is flawed, shake should be applied to the final frame.
            # Let's apply it after rendering everything.
            
        self._render_tunnel()
        self._render_player_and_particles()
        self._render_ui()
        
        if self.screen_shake > 0:
            shake_offset = (random.randint(-8, 8), random.randint(-8, 8))
            temp_surface = pygame.Surface(self.screen.get_size())
            temp_surface.fill(self.COLOR_BG)
            temp_surface.blit(self.screen, shake_offset)
            render_target = temp_surface
        
        arr = pygame.surfarray.array3d(render_target)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score, "steps": self.steps, "lives": self.lives,
            "time_remaining": self.time_remaining,
            "path_progress": self.path_progress / self.path_length if self.path_length > 0 else 0,
            "combo": self.combo, "difficulty": GameEnv.difficulty_level
        }
        
    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    env = GameEnv(render_mode="human_playable")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Neon Tunnel Racer")
    
    terminated = False
    action = [0, 0, 0] 

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

        if not terminated:
            keys = pygame.key.get_pressed()
            action[0] = 0
            if keys[pygame.K_UP] or keys[pygame.K_w]: action[0] = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: action[0] = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: action[0] = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: action[0] = 4
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Level Reached: {info['difficulty']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False