import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import math
from collections import deque
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your city's skyline by placing temporal portals to trap falling enemies. "
        "Use the time rewind ability to send them back to the start."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. "
        "Press space to place a portal and shift to activate the time rewind."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 2000
    DEFENSE_LINE_Y = 350
    GRID_COLS, GRID_ROWS = 16, 10

    # Colors (Dracula Theme)
    COLOR_BG = (40, 42, 54)
    COLOR_GRID = (68, 71, 90)
    COLOR_PLAYER = (241, 250, 140)
    COLOR_STRUCTURE = (80, 250, 123)
    COLOR_STRUCTURE_DMG = (255, 184, 108)
    COLOR_STRUCTURE_DEAD = (255, 85, 85)
    COLOR_ENEMY = (255, 85, 85)
    COLOR_PORTAL = (139, 233, 253)
    COLOR_REWIND_FX = (189, 147, 249)
    COLOR_TEXT = (248, 248, 242)
    COLOR_PARTICLE_EXPLOSION = (255, 121, 198)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Exact spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Grid cell dimensions
        self.grid_cell_width = self.SCREEN_WIDTH // self.GRID_COLS
        self.grid_cell_height = (self.DEFENSE_LINE_Y - 50) // self.GRID_ROWS

        # Initialize state variables to be populated in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.structures = []
        self.enemies = []
        self.portals = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.last_space_held = False
        self.last_shift_held = False
        self.base_enemy_spawn_rate = 0.02
        self.base_enemy_speed = 1.0
        self.time_rewind_effect_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Structures setup
        self.structures = []
        num_structures = 4
        structure_width = 50
        structure_height = 20
        total_width = num_structures * structure_width + (num_structures - 1) * 30
        start_x = (self.SCREEN_WIDTH - total_width) / 2
        for i in range(num_structures):
            x = start_x + i * (structure_width + 30)
            self.structures.append({
                "rect": pygame.Rect(x, self.DEFENSE_LINE_Y, structure_width, structure_height),
                "health": 100,
                "max_health": 100,
                "flicker_timer": 0
            })

        self.enemies = []
        self.portals = []
        self.particles = []

        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]

        self.last_space_held = False
        self.last_shift_held = False

        self.base_enemy_spawn_rate = 0.02
        self.base_enemy_speed = 1.0

        self.time_rewind_effect_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0

        # --- 1. Handle Input & Player Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        self._handle_input(movement, space_pressed, shift_pressed)

        if shift_pressed:
            # // SFX: Time Rewind
            rewound_count = self._activate_time_rewind()
            reward += rewound_count
            if rewound_count > 0:
                self.time_rewind_effect_timer = 15 # frames

        # --- 2. Update Game State ---
        self.steps += 1
        self._update_difficulty()
        self._update_portals()
        self._spawn_enemies()

        reward_info = {"structure_destroyed_penalty": 0}
        self._update_enemies(reward_info)
        reward += reward_info["structure_destroyed_penalty"]

        self._update_particles()

        # --- 3. Calculate Rewards ---
        surviving_structures = sum(1 for s in self.structures if s['health'] > 0)
        reward += surviving_structures * 0.01 # Small reward for surviving
        self.score += reward

        # --- 4. Check Termination ---
        terminated = self._check_termination_conditions()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and not truncated and all(s['health'] <= 0 for s in self.structures): # lost
            pass
        elif truncated and surviving_structures > 0: # won
            reward += 100 # Victory bonus
            self.score += 100

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- State Update Helpers ---

    def _handle_input(self, movement, space_pressed, shift_pressed):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        if movement == 2: self.cursor_pos[1] += 1  # Down
        if movement == 3: self.cursor_pos[0] -= 1  # Left
        if movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        if space_pressed:
            # // SFX: Portal Place
            portal_exists = any(p['grid_pos'] == self.cursor_pos for p in self.portals)
            if not portal_exists:
                world_x = (self.cursor_pos[0] + 0.5) * self.grid_cell_width
                world_y = 50 + (self.cursor_pos[1] + 0.5) * self.grid_cell_height
                self.portals.append({
                    "pos": pygame.Vector2(world_x, world_y),
                    "grid_pos": list(self.cursor_pos),
                    "radius": 20,
                    "lifetime": 300, # 10 seconds at 30fps
                    "spawn_anim": 1.0
                })

    def _activate_time_rewind(self):
        rewound_count = 0
        for enemy in self.enemies:
            if enemy.get("marked_for_rewind", False):
                enemy["pos"] = enemy["rewind_pos"]
                enemy["history"].clear()
                enemy["history"].append(enemy["pos"].copy())
                enemy["marked_for_rewind"] = False
                rewound_count += 1
                self._create_explosion(enemy["pos"], 10, self.COLOR_REWIND_FX, 0.5)
        return rewound_count

    def _update_difficulty(self):
        self.base_enemy_spawn_rate = 0.02 + (self.steps / 100) * 0.001
        self.base_enemy_speed = 1.0 + (self.steps / 200) * 0.02

    def _update_portals(self):
        for p in self.portals:
            p['lifetime'] -= 1
            p['spawn_anim'] = max(0, p['spawn_anim'] - 0.05)
        self.portals = [p for p in self.portals if p['lifetime'] > 0]

    def _spawn_enemies(self):
        if self.np_random.random() < self.base_enemy_spawn_rate:
            x = self.np_random.uniform(20, self.SCREEN_WIDTH - 20)
            speed = self.base_enemy_speed * self.np_random.uniform(0.8, 1.5)

            enemy_type = 0 # Triangle
            if self.steps > 500:
                enemy_type = self.np_random.choice([0, 1]) # Add Squares
            if self.steps > 1000:
                enemy_type = self.np_random.choice([0, 1, 2]) # Add Circles

            size = 10
            if enemy_type == 1: size = 12 # Square is bigger
            if enemy_type == 2: speed *= 1.2 # Circle is faster

            pos = pygame.Vector2(x, -size)
            self.enemies.append({
                "pos": pos,
                "vel": pygame.Vector2(0, speed),
                "size": size,
                "type": enemy_type,
                "history": deque([pos.copy()], maxlen=150), # 5 seconds of history
                "marked_for_rewind": False
            })

    def _update_enemies(self, reward_info):
        for enemy in self.enemies[:]:
            enemy["pos"] += enemy["vel"]
            if self.steps % 2 == 0: # Store history at half rate
                enemy["history"].append(enemy["pos"].copy())

            enemy_rect = pygame.Rect(enemy["pos"].x - enemy["size"], enemy["pos"].y - enemy["size"], enemy["size"]*2, enemy["size"]*2)

            # Check portal collision
            if not enemy["marked_for_rewind"]:
                for portal in self.portals:
                    if portal['spawn_anim'] <= 0 and enemy["pos"].distance_to(portal["pos"]) < portal["radius"]:
                        enemy["marked_for_rewind"] = True
                        enemy["rewind_pos"] = enemy["history"][0]
                        # // SFX: Enemy Marked
                        break

            # Check structure collision
            for struct in self.structures:
                if struct['health'] > 0 and struct['rect'].colliderect(enemy_rect):
                    # // SFX: Explosion
                    self._create_explosion(enemy["pos"], 30, self.COLOR_PARTICLE_EXPLOSION)
                    struct['health'] -= 25
                    struct['flicker_timer'] = 10
                    assert struct['health'] <= struct['max_health']
                    if struct['health'] <= 0:
                        reward_info["structure_destroyed_penalty"] -= 10
                    self.enemies.remove(enemy)
                    break
            else: # if no break from loop
                if enemy["pos"].y > self.SCREEN_HEIGHT:
                    self.enemies.remove(enemy)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['vel'] *= 0.98 # Damping
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _check_termination_conditions(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if all(s['health'] <= 0 for s in self.structures):
            return True
        return False

    # --- Rendering Helpers ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Grid
        for i in range(self.GRID_COLS + 1):
            x = i * self.grid_cell_width
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 50), (x, self.DEFENSE_LINE_Y - 1))
        for i in range(self.GRID_ROWS + 1):
            y = 50 + i * self.grid_cell_height
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Defense Line
        pygame.draw.line(self.screen, self.COLOR_STRUCTURE, (0, self.DEFENSE_LINE_Y), (self.SCREEN_WIDTH, self.DEFENSE_LINE_Y), 2)

    def _render_game_elements(self):
        # Portals
        for p in self.portals:
            radius = p['radius'] * (1.0 - p['spawn_anim'])
            self._draw_glowing_circle(self.screen, self.COLOR_PORTAL, p['pos'], radius, 4)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / p['max_lifespan']))))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])), special_flags=pygame.BLEND_RGBA_ADD)

        # Enemies
        for enemy in self.enemies:
            color = self.COLOR_ENEMY
            if enemy.get("marked_for_rewind", False):
                color = self.COLOR_REWIND_FX

            if enemy['type'] == 0: # Triangle
                p1 = (enemy['pos'].x, enemy['pos'].y - enemy['size'])
                p2 = (enemy['pos'].x - enemy['size']*0.866, enemy['pos'].y + enemy['size']*0.5)
                p3 = (enemy['pos'].x + enemy['size']*0.866, enemy['pos'].y + enemy['size']*0.5)
                self._draw_aa_polygon(self.screen, [p1, p2, p3], color)
            elif enemy['type'] == 1: # Square
                size = enemy['size']
                rect = pygame.Rect(enemy['pos'].x - size/2, enemy['pos'].y - size/2, size, size)
                pygame.draw.rect(self.screen, color, rect)
            elif enemy['type'] == 2: # Circle
                pygame.gfxdraw.aacircle(self.screen, int(enemy['pos'].x), int(enemy['pos'].y), int(enemy['size']), color)
                pygame.gfxdraw.filled_circle(self.screen, int(enemy['pos'].x), int(enemy['pos'].y), int(enemy['size']), color)

        # Structures
        for s in self.structures:
            if s['flicker_timer'] > 0:
                s['flicker_timer'] -= 1
                if s['flicker_timer'] % 4 < 2: continue # Skip drawing for flicker effect

            color = self.COLOR_STRUCTURE
            if s['health'] <= 0: color = self.COLOR_STRUCTURE_DEAD
            elif s['health'] < s['max_health']: color = self.COLOR_STRUCTURE_DMG

            pygame.draw.rect(self.screen, color, s['rect'], border_radius=3)

            # Health bar
            if s['health'] > 0:
                bar_bg_rect = s['rect'].copy()
                bar_bg_rect.y += s['rect'].height + 5
                bar_bg_rect.height = 5
                pygame.draw.rect(self.screen, self.COLOR_GRID, bar_bg_rect, border_radius=2)

                health_ratio = s['health'] / s['max_health']
                bar_fg_rect = bar_bg_rect.copy()
                bar_fg_rect.width = int(bar_bg_rect.width * health_ratio)
                pygame.draw.rect(self.screen, self.COLOR_STRUCTURE, bar_fg_rect, border_radius=2)

        # Portal Cursor
        cursor_world_x = (self.cursor_pos[0] + 0.5) * self.grid_cell_width
        cursor_world_y = 50 + (self.cursor_pos[1] + 0.5) * self.grid_cell_height
        self._draw_glowing_circle(self.screen, self.COLOR_PLAYER, (cursor_world_x, cursor_world_y), 15, 3)

        # Time Rewind Effect
        if self.time_rewind_effect_timer > 0:
            self.time_rewind_effect_timer -= 1
            alpha = 100 * (self.time_rewind_effect_timer / 15)
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_REWIND_FX + (int(alpha),))
            self.screen.blit(overlay, (0,0))

    def _render_ui(self):
        time_text = f"TIME: {self.MAX_STEPS - self.steps}"
        score_text = f"SCORE: {int(self.score)}"

        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)

        self.screen.blit(time_surf, (10, 10))
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))

    # --- Utility Helpers ---
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "surviving_structures": sum(1 for s in self.structures if s['health'] > 0)
        }

    def _create_explosion(self, pos, count, color, speed_multiplier=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) * speed_multiplier
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "size": self.np_random.integers(1, 4),
                "color": color
            })

    def _draw_glowing_circle(self, surface, color, center, radius, width):
        if radius <= 0: return
        center_int = (int(center[0]), int(center[1]))

        for i in range(width):
            alpha = 150 * (1 - i / width)
            # Create a temporary surface for blending
            temp_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(temp_surf, center_int[0], center_int[1], int(radius + i), color + (int(alpha),))
            surface.blit(temp_surf, (0,0))


    def _draw_aa_polygon(self, surface, points, color):
        try:
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
        except (ValueError, TypeError): # Handles cases where points are invalid
            pass

    def close(self):
        pygame.quit()


# Example usage for testing
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()

    # Override screen for display
    env.screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Skyline Defense")

    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default: no-op
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4

        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render to display by getting the observation and blitting it
        rendered_obs = env._get_observation()
        # The observation is (H, W, C) but pygame wants (W, H, C) for surfarray.
        # The internal `_get_observation` already does the transpose, so we need to reverse it for display
        surf_to_display = pygame.surfarray.make_surface(np.transpose(rendered_obs, (1, 0, 2)))
        env.screen.blit(surf_to_display, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS for smooth visuals

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    env.close()