import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:50:23.227898
# Source Brief: brief_01313.md
# Brief Index: 1313
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Navigate a shattered prism world as a light-bending creature, using gravity flips
    and light refraction to evade geometrically distorted enemies and escape the
    fractured dimension.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a shattered prism world as a light-bending creature. Use gravity flips and light refraction to evade enemies and reach the exit."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to flip gravity. Hold shift to refract light and become invisible to enemies."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    FPS = 30

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (180, 220, 255)
    COLOR_REFRACT = (200, 150, 255)
    COLOR_REFRACT_GLOW = (150, 100, 200)
    COLOR_ENEMY_CALM = (0, 150, 255)
    COLOR_ENEMY_ALERT = (255, 50, 50)
    COLOR_PRISM = (70, 60, 90)
    COLOR_PRISM_OUTLINE = (100, 90, 120)
    COLOR_EXIT = (100, 255, 150)
    COLOR_TEXT = (220, 220, 220)

    # Player settings
    PLAYER_SIZE = 12
    PLAYER_ACCEL = 0.8
    PLAYER_FRICTION = 0.92
    
    # Game mechanics
    REFRACTION_MAX_ENERGY = 100
    REFRACTION_DRAIN_RATE = 1.5
    REFRACTION_RECHARGE_RATE = 0.5
    ENEMY_SPEED_BASE = 1.0
    ENEMY_DETECTION_RADIUS = 60
    ENEMY_SPEED_INCREASE_INTERVAL = 500
    ENEMY_SPEED_INCREASE_AMOUNT = 0.05

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
        self.font = pygame.font.SysFont("monospace", 18, bold=True)

        self.gravity_vectors = [
            np.array([0, 1]),   # Down
            np.array([0, -1]),  # Up
            np.array([-1, 0]),  # Left
            np.array([1, 0]),   # Right
        ]
        
        # This will be initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.gravity_dir = None
        self.refraction_energy = None
        self.is_refracted = None
        self.last_space_state = None
        self.enemies = None
        self.prisms = None
        self.exit_pos = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.dist_to_exit = None
        self.particles = None
        self.gravity_flip_effect = None
        self.detection_effect_alpha = None
        self.enemy_base_speed = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = np.array([100.0, 100.0])
        self.player_vel = np.array([0.0, 0.0])
        self.gravity_dir = 0
        
        self.refraction_energy = self.REFRACTION_MAX_ENERGY
        self.is_refracted = False
        self.last_space_state = 0
        
        self.exit_pos = np.array([self.SCREEN_WIDTH - 80.0, self.SCREEN_HEIGHT - 80.0])
        
        self.prisms = self._generate_prisms()
        self.enemies = self._generate_enemies()
        self.enemy_base_speed = self.ENEMY_SPEED_BASE

        self.particles = []
        self.gravity_flip_effect = []
        self.detection_effect_alpha = 0

        self.dist_to_exit = np.linalg.norm(self.player_pos - self.exit_pos)

        return self._get_observation(), self._get_info()

    def _generate_prisms(self):
        prisms = [
            pygame.Rect(200, 150, 240, 30),
            pygame.Rect(100, 250, 30, 120),
            pygame.Rect(450, 50, 30, 150),
        ]
        # Ensure player start is not inside a prism
        while any(p.collidepoint(self.player_pos) for p in prisms):
            self.player_pos[0] += 20
        # Ensure exit is not inside a prism
        while any(p.collidepoint(self.exit_pos) for p in prisms):
            self.exit_pos[0] -= 20
        return prisms

    def _generate_enemies(self):
        enemies = [
            {
                "path": [np.array([50.0, 50.0]), np.array([590.0, 50.0])],
                "path_idx": 0, "state": "calm", "pos": np.array([50.0, 50.0]),
            },
            {
                "path": [np.array([590.0, 350.0]), np.array([50.0, 350.0])],
                "path_idx": 0, "state": "calm", "pos": np.array([590.0, 350.0]),
            },
            {
                "path": [np.array([320.0, 80.0]), np.array([320.0, 320.0])],
                "path_idx": 0, "state": "calm", "pos": np.array([320.0, 80.0]),
            },
        ]
        return enemies

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Gravity Flip (on press)
        gravity_flipped_this_step = False
        if space_held and not self.last_space_state:
            self.gravity_dir = (self.gravity_dir + 1) % 4
            self.gravity_flip_effect.append({"pos": self.player_pos.copy(), "radius": 0, "alpha": 255})
            gravity_flipped_this_step = True
            # sfx: gravity_shift.wav
        self.last_space_state = space_held

        # Refraction
        if shift_held and self.refraction_energy > 0:
            self.is_refracted = True
            self.refraction_energy = max(0, self.refraction_energy - self.REFRACTION_DRAIN_RATE)
            if self.refraction_energy == 0:
                self.is_refracted = False
                # sfx: refraction_end.wav
        else:
            self.is_refracted = False
            self.refraction_energy = min(self.REFRACTION_MAX_ENERGY, self.refraction_energy + self.REFRACTION_RECHARGE_RATE)
        
        # --- Player Movement ---
        gravity_vec = self.gravity_vectors[self.gravity_dir]
        up_vec = -gravity_vec
        right_vec = np.array([gravity_vec[1], -gravity_vec[0]])
        
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec = up_vec       # Up
        elif movement == 2: move_vec = gravity_vec  # Down
        elif movement == 3: move_vec = -right_vec # Left
        elif movement == 4: move_vec = right_vec  # Right
        
        if np.any(move_vec):
            self.player_vel += move_vec * self.PLAYER_ACCEL
            # Add thrust particles
            if self.steps % 2 == 0:
                p_vel = -move_vec * self.np_random.uniform(1, 3) + self.np_random.uniform(-0.5, 0.5, 2)
                self.particles.append({
                    'pos': self.player_pos.copy(), 'vel': p_vel, 'radius': self.np_random.uniform(2, 4),
                    'lifespan': self.np_random.integers(10, 20), 'color': self.COLOR_PLAYER_GLOW
                })

        # Apply gravity and friction
        self.player_vel += gravity_vec * 0.2
        self.player_vel *= self.PLAYER_FRICTION
        self.player_pos += self.player_vel

        # --- Collisions and Boundaries ---
        self._handle_collisions()

        # --- Enemy Logic ---
        detection_this_step = False
        for enemy in self.enemies:
            # Movement
            target_pos = enemy["path"][enemy["path_idx"]]
            direction = target_pos - enemy["pos"]
            dist = np.linalg.norm(direction)
            
            speed = self.enemy_base_speed
            if enemy["state"] == "alerted":
                speed *= 2.0
            
            if dist < speed:
                enemy["pos"] = target_pos
                enemy["path_idx"] = (enemy["path_idx"] + 1) % len(enemy["path"])
            else:
                enemy["pos"] += (direction / dist) * speed
            
            # Detection
            dist_to_player = np.linalg.norm(self.player_pos - enemy["pos"])
            if dist_to_player < self.ENEMY_DETECTION_RADIUS:
                if not self.is_refracted:
                    enemy["state"] = "alerted"
                    detection_this_step = True
                    # sfx: enemy_alert.wav
                else: # In radius but refracted
                    reward -= 0.5

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % self.ENEMY_SPEED_INCREASE_INTERVAL == 0:
            self.enemy_base_speed += self.ENEMY_SPEED_INCREASE_AMOUNT

        # --- Update Game State & Rewards ---
        if detection_this_step:
            self.game_over = True
            reward -= 10
            self.detection_effect_alpha = 255
        
        if gravity_flipped_this_step and not detection_this_step:
            reward += 5

        # Distance-based reward
        new_dist_to_exit = np.linalg.norm(self.player_pos - self.exit_pos)
        reward += (self.dist_to_exit - new_dist_to_exit) * 0.1
        self.dist_to_exit = new_dist_to_exit

        # --- Termination Conditions ---
        terminated = self.game_over
        truncated = False
        if np.linalg.norm(self.player_pos - self.exit_pos) < self.PLAYER_SIZE + 15:
            reward += 100
            terminated = True
            # sfx: win_portal.wav
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            truncated = True # Conventionally, timeout is a truncation
        
        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_collisions(self):
        # Screen boundaries
        if self.player_pos[0] < self.PLAYER_SIZE:
            self.player_pos[0] = self.PLAYER_SIZE
            self.player_vel[0] *= -0.5
        if self.player_pos[0] > self.SCREEN_WIDTH - self.PLAYER_SIZE:
            self.player_pos[0] = self.SCREEN_WIDTH - self.PLAYER_SIZE
            self.player_vel[0] *= -0.5
        if self.player_pos[1] < self.PLAYER_SIZE:
            self.player_pos[1] = self.PLAYER_SIZE
            self.player_vel[1] *= -0.5
        if self.player_pos[1] > self.SCREEN_HEIGHT - self.PLAYER_SIZE:
            self.player_pos[1] = self.SCREEN_HEIGHT - self.PLAYER_SIZE
            self.player_vel[1] *= -0.5
        
        # Prism collisions
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE, self.player_pos[1] - self.PLAYER_SIZE, self.PLAYER_SIZE*2, self.PLAYER_SIZE*2)
        for prism in self.prisms:
            if player_rect.colliderect(prism):
                # Find overlap and push player out
                overlap_x = min(player_rect.right, prism.right) - max(player_rect.left, prism.left)
                overlap_y = min(player_rect.bottom, prism.bottom) - max(player_rect.top, prism.top)

                if overlap_x < overlap_y:
                    if self.player_pos[0] < prism.centerx: # came from left
                        self.player_pos[0] -= overlap_x
                    else: # came from right
                        self.player_pos[0] += overlap_x
                    self.player_vel[0] *= -0.5
                else:
                    if self.player_pos[1] < prism.centery: # came from top
                        self.player_pos[1] -= overlap_y
                    else: # came from bottom
                        self.player_pos[1] += overlap_y
                    self.player_vel[1] *= -0.5

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_effects()
        self._render_prisms()
        self._render_exit()
        self._render_enemies()
        self._render_particles()
        self._render_player()
        self._render_foreground_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "refraction_energy": self.refraction_energy,
            "gravity_direction": self.gravity_dir,
            "distance_to_exit": self.dist_to_exit,
        }

    # --- Rendering Methods ---
    def _render_background_effects(self):
        # Gravity direction indicator
        indicator_size = 20
        margin = 30
        center = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        dirs = [
            (center[0], self.SCREEN_HEIGHT - margin), # Down
            (center[0], margin), # Up
            (margin, center[1]), # Left
            (self.SCREEN_WIDTH - margin, center[1]), # Right
        ]
        points = [
            (dirs[self.gravity_dir][0] - indicator_size, dirs[self.gravity_dir][1] - indicator_size),
            (dirs[self.gravity_dir][0] + indicator_size, dirs[self.gravity_dir][1] - indicator_size),
            (dirs[self.gravity_dir][0] + indicator_size, dirs[self.gravity_dir][1] + indicator_size),
            (dirs[self.gravity_dir][0] - indicator_size, dirs[self.gravity_dir][1] + indicator_size),
        ]
        # Rotate points to indicate gravity
        angle = self.gravity_dir * math.pi / 2
        rot_matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        
        arrow_pts = np.array([[-10, -5], [0, 5], [10, -5]])
        rotated_arrow = (arrow_pts @ rot_matrix) + dirs[self.gravity_dir]
        
        pygame.gfxdraw.aapolygon(self.screen, rotated_arrow, (50, 40, 70))
        pygame.gfxdraw.filled_polygon(self.screen, rotated_arrow, (40, 30, 60))

    def _render_prisms(self):
        for prism in self.prisms:
            pygame.draw.rect(self.screen, self.COLOR_PRISM, prism)
            pygame.draw.rect(self.screen, self.COLOR_PRISM_OUTLINE, prism, 2)

    def _render_exit(self):
        pos = (int(self.exit_pos[0]), int(self.exit_pos[1]))
        t = self.steps * 0.1
        for i in range(5):
            radius = 15 + i * 3 + math.sin(t + i) * 2
            alpha = 100 - i * 15
            color = (*self.COLOR_EXIT, alpha)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(radius), color)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            size = 10
            t = self.steps * 0.2
            
            color = self.COLOR_ENEMY_CALM if enemy["state"] == "calm" else self.COLOR_ENEMY_ALERT
            
            # Pulsating outline
            pulse_radius = size + 2 + abs(math.sin(t)) * 3
            pulse_color = (*color, 50)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(pulse_radius), pulse_color)

            # Main body (triangle)
            points = [
                (pos[0], pos[1] - size),
                (pos[0] - size, pos[1] + size),
                (pos[0] + size, pos[1] + size)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

            # Detection radius if player is not refracted
            if not self.is_refracted:
                radius_color = (*color, 20)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ENEMY_DETECTION_RADIUS, radius_color)


    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        size = self.PLAYER_SIZE
        
        color = self.COLOR_PLAYER if not self.is_refracted else self.COLOR_REFRACT
        glow_color = self.COLOR_PLAYER_GLOW if not self.is_refracted else self.COLOR_REFRACT_GLOW
        
        # Refraction effect
        alpha_mod = 1.0
        if self.is_refracted:
            alpha_mod = 0.5 + 0.5 * math.sin(self.steps * 0.3)
            size *= 0.8 + 0.2 * math.sin(self.steps * 0.3)

        # Glow
        for i in range(3):
            glow_rad = int(size * (1.5 + i * 0.5 + 0.2 * math.sin(self.steps * 0.1 + i)))
            glow_alpha = int((50 - i * 15) * alpha_mod)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_rad, (*glow_color, glow_alpha))

        # Core shape (diamond)
        points = [
            (pos[0], pos[1] - size),
            (pos[0] + size * 0.8, pos[1]),
            (pos[0], pos[1] + size),
            (pos[0] - size * 0.8, pos[1])
        ]
        main_color = (*color, int(255 * alpha_mod))
        pygame.gfxdraw.aapolygon(self.screen, points, main_color)
        pygame.gfxdraw.filled_polygon(self.screen, points, main_color)
    
    def _render_particles(self):
        # Update and draw particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] *= 0.95
            if p['lifespan'] <= 0 or p['radius'] < 0.5:
                self.particles.remove(p)
            else:
                color = (*p['color'], int(p['lifespan'] * 12))
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

    def _render_foreground_effects(self):
        # Gravity flip shockwave
        for effect in self.gravity_flip_effect[:]:
            effect['radius'] += 15
            effect['alpha'] -= 15
            if effect['alpha'] <= 0:
                self.gravity_flip_effect.remove(effect)
            else:
                color = (*self.COLOR_PLAYER_GLOW, effect['alpha'])
                pygame.gfxdraw.aacircle(self.screen, int(effect['pos'][0]), int(effect['pos'][1]), int(effect['radius']), color)
                pygame.gfxdraw.aacircle(self.screen, int(effect['pos'][0]), int(effect['pos'][1]), int(effect['radius']-2), color)

        # Detection distortion/flash
        if self.detection_effect_alpha > 0:
            self.detection_effect_alpha = max(0, self.detection_effect_alpha - 10)
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((*self.COLOR_ENEMY_ALERT, self.detection_effect_alpha))
            # Wave distortion
            for i in range(10):
                y_offset = (i * self.SCREEN_HEIGHT / 10 + self.steps * 5) % self.SCREEN_HEIGHT
                x_offset = math.sin(y_offset / 50 + self.steps * 0.1) * 20
                pygame.draw.line(overlay, (0,0,0,50), (0, y_offset), (self.SCREEN_WIDTH + x_offset, y_offset), 2)
            self.screen.blit(overlay, (0, 0))

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Steps
        steps_text = self.font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        # Refraction energy bar
        bar_width = 150
        bar_height = 12
        bar_x = 10
        bar_y = 35
        energy_ratio = self.refraction_energy / self.REFRACTION_MAX_ENERGY
        
        pygame.draw.rect(self.screen, (50, 50, 80), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_REFRACT, (bar_x, bar_y, int(bar_width * energy_ratio), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Pygame setup for manual play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Prism Break")
    clock = pygame.time.Clock()

    total_reward = 0.0

    while running:
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            terminated = False
            total_reward = 0.0

        # --- Manual Control Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                # Reset on 'r' key press
                terminated = True 
                
        clock.tick(GameEnv.FPS)

    env.close()