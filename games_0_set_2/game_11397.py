import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A neon-drenched, retro-futuristic racing game Gymnasium environment.
    The player controls a vehicle on a closed-loop track, aiming to achieve
    the best lap time by mastering drifting and portal mechanics.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    # --- METADATA ---
    game_description = (
        "A neon-drenched, retro-futuristic racing game. Master drifting and portals "
        "on a closed-loop track to achieve the best lap time."
    )
    user_guide = (
        "Controls: Use ↑ to accelerate, ↓ to brake, and ←→ to turn. "
        "Hold space to drift and press shift to use portals."
    )
    auto_advance = True

    # --- COLORS ---
    COLOR_BG = (10, 0, 25)
    COLOR_TRACK = (60, 0, 120)
    COLOR_TRACK_BORDER = (120, 0, 240)
    COLOR_PLAYER = (255, 0, 128)
    COLOR_PLAYER_GLOW = (255, 100, 200)
    COLOR_PORTAL = (0, 255, 150)
    COLOR_PORTAL_GLOW = (150, 255, 200)
    COLOR_DRIFT_TRAIL = (255, 128, 0)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_ACCENT = (0, 200, 255)
    COLOR_WARNING = (255, 50, 50)

    # --- GAME CONSTANTS ---
    MAX_EPISODE_STEPS = 2000
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30

    # --- PLAYER PHYSICS ---
    ACCELERATION = 0.25
    BRAKING = 0.4
    MAX_SPEED = 6.0
    FRICTION = 0.98
    TURN_SPEED = 0.05
    DRIFT_TURN_MULTIPLIER = 1.8
    DRIFT_FRICTION = 0.99
    DRIFT_ENTRY_SPEED_THRESHOLD = 2.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Game state variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.is_drifting = None
        self.drift_combo = None
        self.particles = None
        self.lap_time = None
        self.portal_cooldown = None
        self.target_lap_time = 60.0  # In seconds
        self.last_checkpoint = None
        self.lap_completed_this_step = None
        self.camera_pos = None
        self.stars = None

        self._generate_track()
        
    def _generate_track(self):
        """Creates the track geometry, checkpoints, and portals."""
        track_radius_x, track_radius_y = 250, 120
        center_x, center_y = self.SCREEN_WIDTH * 1.5, self.SCREEN_HEIGHT * 1.5
        points = 64
        self.track_centerline = []
        for i in range(points):
            angle = (i / points) * 2 * math.pi
            x = center_x + track_radius_x * math.cos(angle)
            y = center_y + track_radius_y * math.sin(angle)
            self.track_centerline.append((x, y))

        self.track_width = 80
        self.walls = []
        for i in range(points):
            p1 = self.track_centerline[i]
            p2 = self.track_centerline[(i + 1) % points]
            self.walls.append((p1, p2))

        # Define checkpoints and start/finish line
        self.checkpoints = []
        num_checkpoints = 4
        for i in range(num_checkpoints):
            index = int(points / num_checkpoints * i)
            p1 = self.track_centerline[index]
            p2 = self.track_centerline[index-1]
            angle = math.atan2(p1[1] - p2[1], p1[0] - p2[0]) + math.pi / 2
            half_width = self.track_width * 1.2
            cp_x1 = p1[0] + half_width * math.cos(angle)
            cp_y1 = p1[1] + half_width * math.sin(angle)
            cp_x2 = p1[0] - half_width * math.cos(angle)
            cp_y2 = p1[1] - half_width * math.sin(angle)
            self.checkpoints.append(((cp_x1, cp_y1), (cp_x2, cp_y2)))

        # Define portals
        portal1_idx = int(points * 0.25)
        portal2_idx = int(points * 0.75)
        self.portals = [
            {'pos': self.track_centerline[portal1_idx], 'link': 1, 'radius': 25, 'activation_effect': 0},
            {'pos': self.track_centerline[portal2_idx], 'link': 0, 'radius': 25, 'activation_effect': 0}
        ]
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        start_pos = self.track_centerline[0]
        self.player_pos = [start_pos[0], start_pos[1]]
        self.player_vel = 0.0
        self.player_angle = math.atan2(
            self.track_centerline[1][1] - start_pos[1],
            self.track_centerline[1][0] - start_pos[0]
        )
        self.is_drifting = False
        self.drift_combo = 0
        self.particles = []
        self.lap_time = 0.0
        self.portal_cooldown = 0
        self.last_checkpoint = len(self.checkpoints) - 1
        self.lap_completed_this_step = False
        self.camera_pos = [self.player_pos[0] - self.SCREEN_WIDTH/2, self.player_pos[1] - self.SCREEN_HEIGHT/2]
        
        star_x = self.np_random.integers(0, self.SCREEN_WIDTH * 3, size=200)
        star_y = self.np_random.integers(0, self.SCREEN_HEIGHT * 3, size=200)
        star_size = self.np_random.integers(1, 4, size=200)
        self.stars = [[star_x[i], star_y[i], star_size[i]] for i in range(200)]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.lap_time += 1 / self.TARGET_FPS
        self.lap_completed_this_step = False
        
        # --- 1. Handle Input & Physics ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input_and_physics(movement, space_held)

        # --- 2. Calculate Rewards and Check for Events ---
        reward = 0.0
        reward += 0.1 # Survival reward

        if self.is_drifting:
            reward += 0.5
            self.drift_combo += 1
        else:
            self.drift_combo = 0
            
        reward += self._handle_portals(shift_held)
        reward += self._check_checkpoints()
        
        # --- 3. Check for Termination & Truncation ---
        terminated = False
        truncated = False
        
        if self._check_collision():
            reward = -100.0
            terminated = True
            self.game_over = True

        if self.lap_completed_this_step:
            if self.lap_time < self.target_lap_time:
                time_bonus = 50 + 50 * max(0, (self.target_lap_time - self.lap_time) / self.target_lap_time)
                reward += time_bonus
            else:
                reward += 5.0
            terminated = True
            self.game_over = True

        if self.lap_time > self.target_lap_time * 1.5:
            reward = -10.0
            terminated = True
            self.game_over = True
            
        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input_and_physics(self, movement, space_held):
        can_drift = space_held and abs(self.player_vel) > self.DRIFT_ENTRY_SPEED_THRESHOLD
        if can_drift and not self.is_drifting:
            self.is_drifting = True
        elif not space_held:
            self.is_drifting = False

        friction = self.DRIFT_FRICTION if self.is_drifting else self.FRICTION
        self.player_vel *= friction

        if movement == 1: # Accelerate
            self.player_vel += self.ACCELERATION
        elif movement == 2: # Brake
            self.player_vel = max(0, self.player_vel - self.BRAKING)
        
        self.player_vel = np.clip(self.player_vel, 0, self.MAX_SPEED)

        if self.player_vel > 0.1:
            turn_rate = self.TURN_SPEED
            if self.is_drifting:
                turn_rate *= self.DRIFT_TURN_MULTIPLIER
            
            if movement == 3: # Left
                self.player_angle -= turn_rate
            elif movement == 4: # Right
                self.player_angle += turn_rate
        
        self.player_pos[0] += self.player_vel * math.cos(self.player_angle)
        self.player_pos[1] += self.player_vel * math.sin(self.player_angle)

        self._update_particles(can_drift)

    def _update_particles(self, is_drifting):
        if is_drifting:
            for _ in range(2):
                offset_angle = self.player_angle + math.pi + (self.np_random.random() - 0.5) * 0.5
                offset_radius = 10
                pos = [
                    self.player_pos[0] + offset_radius * math.cos(offset_angle),
                    self.player_pos[1] + offset_radius * math.sin(offset_angle)
                ]
                vel = [(self.np_random.random() - 0.5) * 0.5, (self.np_random.random() - 0.5) * 0.5]
                life = self.np_random.integers(15, 26)
                self.particles.append({'pos': pos, 'vel': vel, 'life': life, 'max_life': life})

        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _handle_portals(self, shift_held):
        if self.portal_cooldown > 0:
            self.portal_cooldown -= 1
        
        for portal in self.portals:
            if portal['activation_effect'] > 0:
                portal['activation_effect'] -= 1

        if shift_held and self.portal_cooldown == 0:
            for i, portal in enumerate(self.portals):
                dist = math.hypot(self.player_pos[0] - portal['pos'][0], self.player_pos[1] - portal['pos'][1])
                if dist < portal['radius'] + 10:
                    linked_portal = self.portals[portal['link']]
                    self.player_pos = list(linked_portal['pos'])
                    
                    p_idx = self.track_centerline.index(linked_portal['pos'])
                    p_next = self.track_centerline[(p_idx + 1) % len(self.track_centerline)]
                    self.player_angle = math.atan2(p_next[1] - linked_portal['pos'][1], p_next[0] - linked_portal['pos'][0])

                    self.portal_cooldown = 60
                    portal['activation_effect'] = 15
                    linked_portal['activation_effect'] = 15
                    return 1.0
        return 0.0

    def _check_collision(self):
        min_dist_sq = float('inf')
        for i in range(len(self.track_centerline)):
            p1 = self.track_centerline[i]
            p2 = self.track_centerline[(i + 1) % len(self.track_centerline)]

            px, py = self.player_pos
            x1, y1 = p1
            x2, y2 = p2
            
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0:
                dist_sq_segment = (px - x1)**2 + (py - y1)**2
            else:
                dot = (px - x1) * dx + (py - y1) * dy
                len_sq = dx*dx + dy*dy
                t = max(0, min(1, dot / len_sq))
                closest_x, closest_y = x1 + t * dx, y1 + t * dy
                dist_sq_segment = (px - closest_x)**2 + (py - closest_y)**2
            
            if dist_sq_segment < min_dist_sq:
                min_dist_sq = dist_sq_segment

        return min_dist_sq > (self.track_width / 2)**2

    def _check_checkpoints(self):
        p1 = [self.player_pos[0] - self.player_vel * math.cos(self.player_angle),
              self.player_pos[1] - self.player_vel * math.sin(self.player_angle)]
        p2 = self.player_pos
        
        if p1[0] == p2[0] and p1[1] == p2[1]: # No movement
            return 0.0

        next_checkpoint_idx = (self.last_checkpoint + 1) % len(self.checkpoints)
        cp = self.checkpoints[next_checkpoint_idx]
        
        if self._line_intersect(p1, p2, cp[0], cp[1]):
            self.last_checkpoint = next_checkpoint_idx
            if next_checkpoint_idx == 0:
                self.lap_completed_this_step = True
                return 5.0
            else:
                return 0.5
        return 0.0

    def _line_intersect(self, p1, p2, p3, p4):
        den = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
        if den == 0:
            return False
        t_num = (p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])
        u_num = -((p1[0] - p2[0]) * (p1[1] - p3[1]) - (p1[1] - p2[1]) * (p1[0] - p3[0]))
        t = t_num / den
        u = u_num / den
        return 0 < t < 1 and 0 < u < 1

    def _get_observation(self):
        target_cam_x = self.player_pos[0] - self.SCREEN_WIDTH / 2 + 150 * math.cos(self.player_angle)
        target_cam_y = self.player_pos[1] - self.SCREEN_HEIGHT / 2 + 150 * math.sin(self.player_angle)
        self.camera_pos[0] += (target_cam_x - self.camera_pos[0]) * 0.1
        self.camera_pos[1] += (target_cam_y - self.camera_pos[1]) * 0.1
        cam_offset = (int(self.camera_pos[0]), int(self.camera_pos[1]))

        self.screen.fill(self.COLOR_BG)
        self._render_stars(cam_offset)
        self._render_track(cam_offset)
        self._render_portals(cam_offset)
        self._render_particles(cam_offset)
        self._render_player(cam_offset)
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_stars(self, offset):
        for star in self.stars:
            x = (star[0] - offset[0] * 0.1) % self.SCREEN_WIDTH
            y = (star[1] - offset[1] * 0.1) % self.SCREEN_HEIGHT
            pygame.draw.circle(self.screen, (200, 200, 255), (x, y), star[2] * 0.5)

    def _render_track(self, offset):
        # Use gfxdraw for antialiasing if available, otherwise regular draw
        try:
            import pygame.gfxdraw
            track_points = [(p[0] - offset[0], p[1] - offset[1]) for p in self.track_centerline]
            pygame.draw.lines(self.screen, self.COLOR_TRACK_BORDER, True, track_points, width=self.track_width + 10)
            pygame.draw.lines(self.screen, self.COLOR_TRACK, True, track_points, width=self.track_width)
        except ImportError:
            pygame.draw.lines(self.screen, self.COLOR_TRACK_BORDER, True, 
                              [(p[0] - offset[0], p[1] - offset[1]) for p in self.track_centerline], 
                              width=self.track_width + 10)
            pygame.draw.lines(self.screen, self.COLOR_TRACK, True,
                              [(p[0] - offset[0], p[1] - offset[1]) for p in self.track_centerline],
                              width=self.track_width)
        
        sf_line = self.checkpoints[0]
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, 
                         (sf_line[0][0] - offset[0], sf_line[0][1] - offset[1]),
                         (sf_line[1][0] - offset[0], sf_line[1][1] - offset[1]), width=5)

    def _render_portals(self, offset):
        try:
            import pygame.gfxdraw
            for portal in self.portals:
                pos = (int(portal['pos'][0] - offset[0]), int(portal['pos'][1] - offset[1]))
                radius = portal['radius']
                if portal['activation_effect'] > 0:
                    glow_radius = radius + portal['activation_effect'] * 2
                    self._draw_glow_circle(pos, glow_radius, self.COLOR_PORTAL_GLOW)
                elif self.portal_cooldown == 0:
                     self._draw_glow_circle(pos, radius + 5, self.COLOR_PORTAL_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_PORTAL)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_PORTAL_GLOW)
        except (ImportError, TypeError): # TypeError can happen in some pygame versions with gfxdraw
             for portal in self.portals:
                pos = (int(portal['pos'][0] - offset[0]), int(portal['pos'][1] - offset[1]))
                pygame.draw.circle(self.screen, self.COLOR_PORTAL, pos, portal['radius'])


    def _render_particles(self, offset):
        for p in self.particles:
            pos = (int(p['pos'][0] - offset[0]), int(p['pos'][1] - offset[1]))
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*self.COLOR_DRIFT_TRAIL, alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (pos[0] - 2, pos[1] - 2))

    def _render_player(self, offset):
        pos = (int(self.player_pos[0] - offset[0]), int(self.player_pos[1] - offset[1]))
        size = 12
        p1 = (pos[0] + size * math.cos(self.player_angle), pos[1] + size * math.sin(self.player_angle))
        p2 = (pos[0] + size * math.cos(self.player_angle + 2.5), pos[1] + size * math.sin(self.player_angle + 2.5))
        p3 = (pos[0] + size * math.cos(self.player_angle - 2.5), pos[1] + size * math.sin(self.player_angle - 2.5))
        points = [p1, p2, p3]

        try:
            import pygame.gfxdraw
            self._draw_glow_poly(points, self.COLOR_PLAYER_GLOW, 20)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        except (ImportError, TypeError):
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)


    def _render_ui(self):
        time_text = self.font_large.render(f"{self.lap_time:.2f}", True, self.COLOR_UI_TEXT)
        target_text = self.font_small.render(f"Target: {self.target_lap_time:.2f}", True, self.COLOR_UI_ACCENT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH // 2 - time_text.get_width() // 2, 10))
        self.screen.blit(target_text, (self.SCREEN_WIDTH // 2 - target_text.get_width() // 2, 50))
        
        speed_text = self.font_large.render(f"{int(self.player_vel * 20)} KPH", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_text, (self.SCREEN_WIDTH - speed_text.get_width() - 20, self.SCREEN_HEIGHT - 60))

        if self.drift_combo > 10:
            combo_text = self.font_small.render(f"Drift x{self.drift_combo}", True, self.COLOR_DRIFT_TRAIL)
            self.screen.blit(combo_text, (20, self.SCREEN_HEIGHT - 40))

        if self.portal_cooldown > 0:
            portal_status_text = self.font_small.render("Portal Offline", True, self.COLOR_WARNING)
        else:
            portal_status_text = self.font_small.render("Portal Ready", True, self.COLOR_PORTAL)
        self.screen.blit(portal_status_text, (self.SCREEN_WIDTH - portal_status_text.get_width() - 20, 20))

    def _draw_glow_circle(self, pos, radius, color):
        try:
            import pygame.gfxdraw
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            for i in range(4):
                alpha = 60 - i * 15
                pygame.gfxdraw.filled_circle(temp_surf, int(radius), int(radius), int(radius - i * 2), (*color, alpha))
            self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))
        except (ImportError, TypeError):
            pass # Glow is optional
        
    def _draw_glow_poly(self, points, color, max_glow_size):
        try:
            min_x = min(p[0] for p in points)
            max_x = max(p[0] for p in points)
            min_y = min(p[1] for p in points)
            max_y = max(p[1] for p in points)
            
            width = max_x - min_x + max_glow_size * 2
            height = max_y - min_y + max_glow_size * 2
            
            temp_surf = pygame.Surface((width, height), pygame.SRCALPHA)
            rel_points = [(p[0] - min_x + max_glow_size, p[1] - min_y + max_glow_size) for p in points]
            
            pygame.draw.polygon(temp_surf, (*color, 20), rel_points, width=max_glow_size)
            pygame.draw.polygon(temp_surf, (*color, 40), rel_points, width=int(max_glow_size*0.6))
            
            self.screen.blit(temp_surf, (min_x - max_glow_size, min_y - max_glow_size))
        except (ImportError, TypeError):
            pass # Glow is optional

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap_time": self.lap_time,
            "drift_combo": self.drift_combo,
            "speed_kph": self.player_vel * 20
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually.
    # It will open a display window.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Neon Drift Racer")
    
    while not done:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(env.metadata["render_fps"])

        if done:
            print(f"Game Over! Final Score: {info['score']:.2f}, Lap Time: {info['lap_time']:.2f}s")
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False

    env.close()