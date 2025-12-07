import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import os
import pygame


# Set SDL to dummy mode for headless operation, which is required for the environment.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

class GameEnv(gym.Env):
    """
    A Gymnasium environment where an agent controls a bouncing ball in a neon tunnel.
    The agent must collect gems, avoid spikes, and can transform into a saw.
    The environment prioritizes visual quality and engaging gameplay.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Navigate a glowing ball through a neon tunnel, collecting gems and avoiding spikes. "
        "Transform into a saw to smash through obstacles."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the ball. Press space to transform into a saw."
    )
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_BALL = (0, 150, 255)
    COLOR_SAW = (255, 50, 50)
    COLOR_GEM = (255, 255, 0)
    COLOR_SPIKE = (255, 0, 255)
    COLOR_TUNNEL = (0, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_PARTICLE_GEM = (255, 255, 150)
    COLOR_PARTICLE_DEATH = (255, 100, 100)

    # Screen Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game Parameters
    FPS = 30
    MAX_STEPS = 2000
    WIN_GEM_COUNT = 15
    MAX_LEVEL = 4
    LEVEL_BASE_LENGTH = 2000 # Pixels

    # Player Physics
    PLAYER_RADIUS = 12
    PLAYER_FORCE = 0.6
    PLAYER_DRAG = 0.98
    PLAYER_BOUNCE_FACTOR = 0.8

    # Saw Mechanic
    SAW_DURATION = 15  # 0.5 seconds at 30 FPS

    # Tunnel Generation
    TUNNEL_SEGMENT_LENGTH = 50
    TUNNEL_MIN_WIDTH = 150
    TUNNEL_MAX_WIDTH = 250
    TUNNEL_ROUGHNESS = 0.4

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.gems_collected = 0

        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.is_saw = False
        self.saw_timer = 0
        self.last_space_held = False
        
        self.camera_pos = np.array([0.0, 0.0])
        
        self.tunnel_path = []
        self.gems = []
        self.spikes = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.gems_collected = 0

        # --- Reset Player ---
        self.player_pos = np.array([200.0, self.SCREEN_HEIGHT / 2.0])
        self.player_vel = np.array([5.0, 0.0]) # Start with forward momentum
        self.is_saw = False
        self.saw_timer = 0
        self.last_space_held = False

        # --- Reset World ---
        self.camera_pos = self.player_pos - np.array([self.SCREEN_WIDTH * 0.2, self.SCREEN_HEIGHT / 2.0])
        self.particles = []
        self.tunnel_path = []
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        # --- Store pre-update state for reward calculation ---
        prev_dist_to_gem = self._get_closest_distance(self.gems)
        prev_dist_to_spike = self._get_closest_distance(self.spikes)

        # --- Update Game Logic ---
        self._update_player()
        self._update_saw()
        self._update_particles()
        self._update_camera()

        # --- Handle Collisions and Events ---
        reward += self._handle_collisions()
        
        # --- Proximity Rewards ---
        new_dist_to_gem = self._get_closest_distance(self.gems)
        new_dist_to_spike = self._get_closest_distance(self.spikes)
        
        if len(self.gems) > 0 and prev_dist_to_gem != float('inf'):
            reward += 0.1 if new_dist_to_gem < prev_dist_to_gem else -0.05
        if len(self.spikes) > 0 and prev_dist_to_spike != float('inf'):
            reward -= 0.1 if new_dist_to_spike < prev_dist_to_spike else 0.0

        # --- Check for Level Completion ---
        if self.tunnel_path and self.player_pos[0] >= self.tunnel_path[-1][0]:
            reward += 50
            self.score += 50
            self.level += 1
            if self.level > self.MAX_LEVEL:
                self.game_over = True
                if self.gems_collected >= self.WIN_GEM_COUNT:
                    reward += 100
                    self.score += 100
            else:
                self._generate_level()

        # --- Check Termination Conditions ---
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            reward -= 20
        
        terminated = self.game_over

        return self._get_observation(), float(reward), terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Movement
        force = np.array([0.0, 0.0])
        if movement == 1:  # Up
            force[1] = -self.PLAYER_FORCE
        elif movement == 2:  # Down
            force[1] = self.PLAYER_FORCE
        elif movement == 3:  # Left
            force[0] = -self.PLAYER_FORCE
        elif movement == 4:  # Right
            force[0] = self.PLAYER_FORCE
        self.player_vel += force

        # Saw Transformation (on press, not hold)
        if space_held and not self.last_space_held and not self.is_saw:
            self.is_saw = True
            self.saw_timer = self.SAW_DURATION
        self.last_space_held = space_held

    def _update_player(self):
        self.player_vel *= self.PLAYER_DRAG
        self.player_pos += self.player_vel

    def _update_saw(self):
        if self.is_saw:
            self.saw_timer -= 1
            if self.saw_timer <= 0:
                self.is_saw = False

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _update_camera(self):
        target_cam_pos = self.player_pos - np.array([self.SCREEN_WIDTH * 0.2, self.SCREEN_HEIGHT / 2])
        self.camera_pos = self.camera_pos * 0.9 + target_cam_pos * 0.1

    def _handle_collisions(self):
        reward = 0
        
        top_y, bottom_y = self._get_tunnel_bounds_at(self.player_pos[0])
        radius = self.PLAYER_RADIUS
        if self.player_pos[1] - radius < top_y:
            self.player_pos[1] = top_y + radius
            self.player_vel[1] *= -self.PLAYER_BOUNCE_FACTOR
        if self.player_pos[1] + radius > bottom_y:
            self.player_pos[1] = bottom_y - radius
            self.player_vel[1] *= -self.PLAYER_BOUNCE_FACTOR

        for gem in self.gems[:]:
            if np.linalg.norm(self.player_pos - gem) < self.PLAYER_RADIUS + 8:
                self.gems.remove(gem)
                self.gems_collected += 1
                reward += 10
                self.score += 10
                self._create_particles(self.player_pos, 20, self.COLOR_PARTICLE_GEM)

        for spike in self.spikes:
            if np.linalg.norm(self.player_pos - spike) < self.PLAYER_RADIUS + 8:
                if not self.is_saw:
                    self.game_over = True
                    reward -= 100
                    self.score -= 100
                    self._create_particles(self.player_pos, 50, self.COLOR_PARTICLE_DEATH, 5)
                    break
        
        return reward

    def _generate_level(self):
        start_x = self.tunnel_path[-1][0] if self.tunnel_path else 0
        if self.level > 1:
            self.player_pos[0] = start_x + 200

        level_length_pixels = self.LEVEL_BASE_LENGTH * (1.5 ** (self.level - 1))
        num_segments = int(level_length_pixels / self.TUNNEL_SEGMENT_LENGTH)
        
        last_y = self.tunnel_path[-1][1] if self.tunnel_path else self.SCREEN_HEIGHT / 2
        last_width = self.tunnel_path[-1][2] if self.tunnel_path else self.TUNNEL_MAX_WIDTH
        
        self.tunnel_path = []
        self.gems = []
        self.spikes = []

        y = last_y
        width = last_width

        for i in range(num_segments):
            x = start_x + i * self.TUNNEL_SEGMENT_LENGTH
            
            y += self.np_random.uniform(-self.TUNNEL_SEGMENT_LENGTH * self.TUNNEL_ROUGHNESS, self.TUNNEL_SEGMENT_LENGTH * self.TUNNEL_ROUGHNESS)
            y = np.clip(y, width / 2, self.SCREEN_HEIGHT - width / 2)
            
            width -= 0.5
            width = np.clip(width, self.TUNNEL_MIN_WIDTH, self.TUNNEL_MAX_WIDTH)
            
            self.tunnel_path.append((x, y, width))

        num_spikes = self.level + 2
        for i in range(1, num_segments - 1):
            segment_x, segment_y, segment_width = self.tunnel_path[i]
            top_bound = segment_y - segment_width / 2
            bottom_bound = segment_y + segment_width / 2
            
            if i % (num_segments // (num_spikes + 1)) == 0 and len(self.spikes) < num_spikes:
                pos_x = segment_x + self.np_random.uniform(-self.TUNNEL_SEGMENT_LENGTH/2, self.TUNNEL_SEGMENT_LENGTH/2)
                pos_y = top_bound + 10 if self.np_random.random() > 0.5 else bottom_bound - 10
                self.spikes.append(np.array([pos_x, pos_y]))

            if i % 5 == 0:
                pos_x = segment_x
                pos_y = self.np_random.uniform(top_bound + 30, bottom_bound - 30)
                self.gems.append(np.array([pos_x, pos_y]))

    def _get_tunnel_bounds_at(self, x_pos):
        if not self.tunnel_path:
            return 0, self.SCREEN_HEIGHT
        
        for i in range(len(self.tunnel_path) - 1):
            x1, y1, w1 = self.tunnel_path[i]
            x2, y2, w2 = self.tunnel_path[i+1]
            if x1 <= x_pos < x2:
                t = (x_pos - x1) / (x2 - x1)
                interp_y = y1 + t * (y2 - y1)
                interp_w = w1 + t * (w2 - w1)
                return interp_y - interp_w / 2, interp_y + interp_w / 2
        
        _, y_last, w_last = self.tunnel_path[-1]
        return y_last - w_last / 2, y_last + w_last / 2

    def _get_closest_distance(self, entities):
        if not entities:
            return float('inf')
        distances = [np.linalg.norm(self.player_pos - e) for e in entities]
        return min(distances)

    def _create_particles(self, pos, count, color, speed_mult=1):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color
            })

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
            "level": self.level,
            "gems_collected": self.gems_collected,
            "player_pos": self.player_pos.tolist(),
            "player_vel": self.player_vel.tolist(),
        }

    def _render_game(self):
        points_top = []
        points_bottom = []
        for x, y, width in self.tunnel_path:
            screen_x = int(x - self.camera_pos[0])
            if 0 <= screen_x <= self.SCREEN_WIDTH:
                points_top.append((screen_x, int(y - width/2 - self.camera_pos[1])))
                points_bottom.append((screen_x, int(y + width/2 - self.camera_pos[1])))
        
        if len(points_top) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_TUNNEL, False, points_top)
            pygame.draw.aalines(self.screen, self.COLOR_TUNNEL, False, points_bottom)

        for spike_pos in self.spikes:
            s_pos = self._world_to_screen(spike_pos)
            if -20 < s_pos[0] < self.SCREEN_WIDTH + 20 and -20 < s_pos[1] < self.SCREEN_HEIGHT + 20:
                self._draw_glowing_circle(s_pos, 8, self.COLOR_SPIKE)

        for gem_pos in self.gems:
            g_pos = self._world_to_screen(gem_pos)
            if -20 < g_pos[0] < self.SCREEN_WIDTH + 20 and -20 < g_pos[1] < self.SCREEN_HEIGHT + 20:
                self._draw_glowing_circle(g_pos, 8, self.COLOR_GEM)

        for p in self.particles:
            p_pos = self._world_to_screen(p['pos'])
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            if 0 <= p_pos[0] < self.SCREEN_WIDTH and 0 <= p_pos[1] < self.SCREEN_HEIGHT:
                 pygame.gfxdraw.pixel(self.screen, int(p_pos[0]), int(p_pos[1]), color)

        p_pos = self._world_to_screen(self.player_pos)
        if -self.PLAYER_RADIUS < p_pos[0] < self.SCREEN_WIDTH + self.PLAYER_RADIUS and \
           -self.PLAYER_RADIUS < p_pos[1] < self.SCREEN_HEIGHT + self.PLAYER_RADIUS:
            if self.is_saw:
                self._draw_glowing_circle(p_pos, self.PLAYER_RADIUS, self.COLOR_SAW)
                for i in range(8):
                    angle = (i / 8.0) * 2 * math.pi + (self.steps * 0.5)
                    start_pos = (p_pos[0] + math.cos(angle) * self.PLAYER_RADIUS, p_pos[1] + math.sin(angle) * self.PLAYER_RADIUS)
                    end_pos = (p_pos[0] + math.cos(angle) * (self.PLAYER_RADIUS + 5), p_pos[1] + math.sin(angle) * (self.PLAYER_RADIUS + 5))
                    pygame.draw.aaline(self.screen, self.COLOR_SAW, start_pos, end_pos)
            else:
                self._draw_glowing_circle(p_pos, self.PLAYER_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        gem_text = self.font.render(f"Gems: {self.gems_collected}/{self.WIN_GEM_COUNT}", True, self.COLOR_TEXT)
        self.screen.blit(gem_text, (10, 10))

        level_text = self.font.render(f"Level: {self.level}/{self.MAX_LEVEL}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 10))
        
        if self.game_over:
            win_condition = self.gems_collected >= self.WIN_GEM_COUNT and self.level > self.MAX_LEVEL
            end_text_str = "YOU WIN!" if win_condition else "GAME OVER"
            end_text = self.font.render(end_text_str, True, self.COLOR_GEM if win_condition else self.COLOR_SAW)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _world_to_screen(self, pos):
        return (
            pos[0] - self.camera_pos[0],
            pos[1] - self.camera_pos[1]
        )

    def _draw_glowing_circle(self, pos, radius, color):
        x, y = int(pos[0]), int(pos[1])
        for i in range(radius, 0, -2):
            alpha = int(150 * (1 - i / radius))
            pygame.gfxdraw.aacircle(self.screen, x, y, i, (*color, alpha))
        
        safe_radius = max(0, radius - 2)
        pygame.gfxdraw.filled_circle(self.screen, x, y, safe_radius, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, safe_radius, color)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    pygame.display.set_caption("Neon Tunnel Bounce")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- Resetting Environment ---")
                obs, info = env.reset(seed=42)
                total_reward = 0

        if terminated or truncated:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}, Score: {info['score']}")
            print("Press 'R' to reset.")
            # Wait for reset
            while True:
                event = pygame.event.wait()
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("--- Resetting Environment ---")
                    obs, info = env.reset(seed=42)
                    total_reward = 0
                    break
            if not running:
                break
        
        clock.tick(GameEnv.FPS)
        
    env.close()