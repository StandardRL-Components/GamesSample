import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:42:25.262818
# Source Brief: brief_01857.md
# Brief Index: 1857
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}
    
    game_description = (
        "Navigate a neon maze, collecting orbs to unlock the exit. Switch between fast, strong, and large forms "
        "to overcome rotating obstacles and escape before time runs out."
    )
    user_guide = "Use arrow keys (↑↓←→) to move. Press space to cycle between fast, strong, and large forms."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Critical Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Game & Visual Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.render_mode = render_mode
        self.FPS = self.metadata["render_fps"]
        self.MAX_TIME_SECONDS = 60
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS
        self.TARGET_ORBS = 100

        # --- Colors (Neon Geometric) ---
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_WALLS = (40, 50, 90)
        self.COLOR_PLATFORM = (180, 80, 255)
        self.COLOR_ORB = (255, 255, 150)
        self.COLOR_EXIT_INACTIVE = (80, 150, 80)
        self.COLOR_EXIT_ACTIVE = (100, 255, 100)
        self.COLOR_TEXT = (220, 220, 240)
        self.PLAYER_COLORS = {
            'fast': (255, 80, 80),   # Red
            'strong': (255, 200, 80), # Yellow
            'large': (80, 150, 255)   # Blue
        }
        
        # --- Player Form Configuration ---
        self.PLAYER_FORMS = ['fast', 'strong', 'large']
        self.PLAYER_ATTRIBUTES = {
            'fast': {'speed': 4.0, 'size': 16},
            'strong': {'speed': 2.5, 'size': 20},
            'large': {'speed': 1.5, 'size': 28}
        }

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 60, bold=True)
        
        self.human_screen = None
        if self.render_mode == "human":
            # This will be properly initialized in render() to avoid issues with pickling
            pass
        
        # --- State Variables (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_remaining = None
        self.orbs_collected = None
        self.player_pos = None
        self.player_form_idx = None
        self.player_speed = None
        self.player_size = None
        self.player_color = None
        self.player_rect = None
        self.walls = None
        self.platforms = None
        self.orbs = None
        self.exit_rect = None
        self.exit_active = None
        self.prev_space_held = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.orbs_collected = 0
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_form_idx = 0  # Start as 'fast'
        self._update_player_form_attributes()
        
        self._generate_maze()
        self._spawn_platforms()
        self._spawn_orbs()
        
        self.exit_rect = pygame.Rect(self.WIDTH - 50, self.HEIGHT / 2 - 25, 30, 50)
        self.exit_active = False
        
        self.prev_space_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Time penalty

        self._handle_form_change(space_held)
        self._handle_player_movement(movement)
        self._update_platforms()

        reward += self._handle_orb_collection()
        reward += self._handle_platform_collisions()
        
        # Check if exit becomes active
        if self.orbs_collected >= self.TARGET_ORBS and not self.exit_active:
            self.exit_active = True
            # sfx_exit_activated

        terminated = False
        truncated = False
        if self.exit_active and self.player_rect.colliderect(self.exit_rect):
            reward += 100
            self.score = 1 # Win
            terminated = True
            # sfx_win_game
        
        if self.time_remaining <= 0 and not terminated:
            reward -= 50
            self.score = -1 # Loss
            terminated = True
            # sfx_lose_game

        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Per Gymnasium API, truncated envs are also terminated

        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_walls()
        self._render_exit()
        self._render_platforms()
        self._render_orbs()
        self._render_player()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "orbs_collected": self.orbs_collected,
            "time_remaining_steps": self.time_remaining,
            "player_form": self.PLAYER_FORMS[self.player_form_idx],
        }

    # --- Helper Methods: Game Logic ---

    def _update_player_form_attributes(self):
        form_key = self.PLAYER_FORMS[self.player_form_idx]
        attributes = self.PLAYER_ATTRIBUTES[form_key]
        self.player_speed = attributes['speed']
        self.player_size = attributes['size']
        self.player_color = self.PLAYER_COLORS[form_key]
        self.player_rect = pygame.Rect(0, 0, self.player_size, self.player_size)
        self.player_rect.center = self.player_pos

    def _handle_form_change(self, space_held):
        if space_held and not self.prev_space_held:
            self.player_form_idx = (self.player_form_idx + 1) % len(self.PLAYER_FORMS)
            self._update_player_form_attributes()
            # sfx_form_change
        self.prev_space_held = space_held

    def _handle_player_movement(self, movement):
        move_vec = np.array([0, 0], dtype=np.float32)
        if movement == 1: move_vec[1] = -1 # Up
        elif movement == 2: move_vec[1] = 1 # Down
        elif movement == 3: move_vec[0] = -1 # Left
        elif movement == 4: move_vec[0] = 1 # Right
        
        if np.linalg.norm(move_vec) > 0:
            move_vec = move_vec / np.linalg.norm(move_vec)
        
        new_pos = self.player_pos + move_vec * self.player_speed
        
        # Boundary checks
        new_pos[0] = np.clip(new_pos[0], self.player_size / 2, self.WIDTH - self.player_size / 2)
        new_pos[1] = np.clip(new_pos[1], self.player_size / 2, self.HEIGHT - self.player_size / 2)
        
        # Wall collision
        new_rect = self.player_rect.copy()
        new_rect.center = (int(new_pos[0]), int(new_pos[1]))
        
        if new_rect.collidelist(self.walls) == -1:
            self.player_pos = new_pos
        
        self.player_rect.center = (int(self.player_pos[0]), int(self.player_pos[1]))

    def _update_platforms(self):
        # Difficulty scaling
        speed_increase = 0.00005 * self.steps
        for p in self.platforms:
            p['angle'] = (p['angle'] + p['base_speed'] + speed_increase) % (2 * math.pi)
            rotated_surface = pygame.transform.rotate(p['surface'], -math.degrees(p['angle']))
            p['rect'] = rotated_surface.get_rect(center=p['center'])

    def _handle_orb_collection(self):
        collected_reward = 0
        for orb in self.orbs[:]:
            if self.player_rect.colliderect(orb['rect']):
                self.orbs.remove(orb)
                self.orbs_collected += 1
                collected_reward += 0.1
                # sfx_orb_collect
        return collected_reward

    def _handle_platform_collisions(self):
        collision_reward = 0
        is_strong_form = self.PLAYER_FORMS[self.player_form_idx] == 'strong'
        
        for p in self.platforms:
            if self.player_rect.colliderect(p['rect']):
                # Check for synchronization (platform is near horizontal/vertical)
                angle_deg = math.degrees(p['angle']) % 180
                sync_tolerance = 10 # degrees
                is_synchronized = (abs(angle_deg) < sync_tolerance or 
                                   abs(angle_deg - 90) < sync_tolerance or
                                   abs(angle_deg - 180) < sync_tolerance)

                if is_synchronized:
                    collision_reward += 5
                    # sfx_sync_success
                elif not is_strong_form:
                    collision_reward -= 1
                    # sfx_platform_hit
        return collision_reward

    def _generate_maze(self):
        self.walls = []
        wall_thickness = 15
        # Borders
        self.walls.append(pygame.Rect(0, 0, self.WIDTH, wall_thickness))
        self.walls.append(pygame.Rect(0, self.HEIGHT - wall_thickness, self.WIDTH, wall_thickness))
        self.walls.append(pygame.Rect(0, 0, wall_thickness, self.HEIGHT))
        self.walls.append(pygame.Rect(self.WIDTH - wall_thickness, 0, wall_thickness, self.HEIGHT))
        # Internal walls
        self.walls.append(pygame.Rect(100, 100, wall_thickness, 200))
        self.walls.append(pygame.Rect(self.WIDTH - 115, 100, wall_thickness, 200))
        self.walls.append(pygame.Rect(200, 0, wall_thickness, 150))
        self.walls.append(pygame.Rect(self.WIDTH - 215, self.HEIGHT - 150, wall_thickness, 150))

    def _spawn_platforms(self):
        self.platforms = []
        platform_data = [
            {'center': (180, 200), 'size': (100, 15), 'speed': 0.02},
            {'center': (self.WIDTH - 180, 200), 'size': (100, 15), 'speed': -0.025},
            {'center': (self.WIDTH / 2, 80), 'size': (150, 10), 'speed': 0.015},
            {'center': (self.WIDTH / 2, self.HEIGHT - 80), 'size': (150, 10), 'speed': -0.01}
        ]
        for data in platform_data:
            surface = pygame.Surface(data['size'], pygame.SRCALPHA)
            surface.fill(self.COLOR_PLATFORM)
            self.platforms.append({
                'center': data['center'],
                'angle': self.np_random.uniform(0, 2 * math.pi),
                'base_speed': data['speed'],
                'surface': surface,
                'rect': surface.get_rect(center=data['center'])
            })

    def _spawn_orbs(self):
        self.orbs = []
        orb_size = 8
        num_orbs_to_spawn = 150 # More than needed
        while len(self.orbs) < num_orbs_to_spawn:
            pos = (self.np_random.integers(20, self.WIDTH - 20),
                   self.np_random.integers(20, self.HEIGHT - 20))
            orb_rect = pygame.Rect(pos[0] - orb_size/2, pos[1] - orb_size/2, orb_size, orb_size)
            if orb_rect.collidelist(self.walls) == -1:
                self.orbs.append({'rect': orb_rect, 'pos': pos})

    # --- Helper Methods: Rendering ---

    def _render_glow_rect(self, surface, color, rect, radius):
        for i in range(radius, 0, -2):
            alpha = 30 - int(i * (30 / radius))
            glow_color = (*color, alpha)
            glow_rect = rect.inflate(i, i)
            pygame.gfxdraw.box(surface, glow_rect, glow_color)

    def _render_walls(self):
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALLS, wall)

    def _render_exit(self):
        color = self.COLOR_EXIT_ACTIVE if self.exit_active else self.COLOR_EXIT_INACTIVE
        if self.exit_active:
            self._render_glow_rect(self.screen, color, self.exit_rect, 20)
        pygame.draw.rect(self.screen, color, self.exit_rect)

    def _render_platforms(self):
        for p in self.platforms:
            rotated_surface = pygame.transform.rotate(p['surface'], -math.degrees(p['angle']))
            rect = rotated_surface.get_rect(center=p['center'])
            self.screen.blit(rotated_surface, rect.topleft)

    def _render_orbs(self):
        for orb in self.orbs:
            pygame.gfxdraw.filled_circle(self.screen, int(orb['pos'][0]), int(orb['pos'][1]), 4, self.COLOR_ORB)
            pygame.gfxdraw.aacircle(self.screen, int(orb['pos'][0]), int(orb['pos'][1]), 4, self.COLOR_ORB)

    def _render_player(self):
        self._render_glow_rect(self.screen, self.player_color, self.player_rect, 25)
        pygame.draw.rect(self.screen, self.player_color, self.player_rect, border_radius=3)

    def _render_ui(self):
        orb_text = self.font_ui.render(f"Orbs: {self.orbs_collected}/{self.TARGET_ORBS}", True, self.COLOR_TEXT)
        self.screen.blit(orb_text, (10, 10))
        
        time_text = self.font_ui.render(f"Time: {self.time_remaining / self.FPS:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        
        form_key = self.PLAYER_FORMS[self.player_form_idx]
        form_text = self.font_ui.render(f"Form: {form_key.upper()}", True, self.PLAYER_COLORS[form_key])
        self.screen.blit(form_text, (self.WIDTH // 2 - form_text.get_width() // 2, 10))

    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        if self.score > 0:
            text = "VICTORY"
            color = self.COLOR_EXIT_ACTIVE
        else:
            text = "TIME UP"
            color = self.PLAYER_COLORS['fast']
        
        game_over_text = self.font_game_over.render(text, True, color)
        text_rect = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        
        overlay.blit(game_over_text, text_rect)
        self.screen.blit(overlay, (0, 0))

    def render(self):
        if self.render_mode == "human":
            if self.human_screen is None:
                pygame.display.init()
                self.human_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            
            frame = self._get_observation()
            # The observation is (H, W, C), but pygame wants (W, H, C) for surfarray
            # Transpose it back
            frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            self.human_screen.blit(frame_surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.FPS)
        return self._get_observation()

    def close(self):
        if self.human_screen:
            pygame.display.quit()
            self.human_screen = None
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    terminated = False
    
    # Manual play loop
    while not terminated:
        movement = 0 # none
        space_held = 0
        shift_held = 0

        # This is for human play, and needs a display.
        # It will fail in a headless environment.
        try:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
            # Check for quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
        except pygame.error as e:
            print(f"Pygame error (expected in headless mode): {e}")
            terminated = True # Exit if we can't get events

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
    
    print("Game Over. Final Info:", info)
    pygame.time.wait(2000) # Wait 2 seconds before closing
    env.close()