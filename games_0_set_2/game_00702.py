import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the placement cursor. "
        "Space to place the selected tower. Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of zombies by strategically placing various towers on the grid."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    TILE_SIZE = 40

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (40, 40, 55)
    COLOR_PATH = (50, 50, 70)
    COLOR_BASE = (0, 150, 50)
    COLOR_BASE_DMG = (200, 50, 50)
    COLOR_ZOMBIE = (220, 40, 40)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_OUTLINE = (10, 10, 10)

    TOWER_SPECS = [
        {"name": "Gatling", "cost": 100, "range": 100, "damage": 5, "fire_rate": 5, "color": (80, 160, 255), "proj_speed": 8},
        {"name": "Cannon", "cost": 250, "range": 150, "damage": 40, "fire_rate": 45, "color": (255, 150, 50), "proj_speed": 6},
        {"name": "Frost", "cost": 150, "range": 80, "damage": 0, "fire_rate": 10, "color": (160, 100, 255), "slow_factor": 0.5},
    ]

    # Game Parameters
    MAX_STEPS = 30000  # Approx 15 mins at 30fps
    MAX_WAVES = 10
    INITIAL_BASE_HEALTH = 100
    INITIAL_RESOURCES = 250
    RESOURCES_PER_WAVE = 150
    WAVE_COOLDOWN_STEPS = 300  # 10 seconds at 30fps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 32)

        self.render_mode = render_mode

        # Initialize state variables
        self.zombies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.path = []

        # This is called at the end of __init__ after all attributes are defined.
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Use Python's built-in random for non-numpy randomness
            random.seed(seed)
            # We don't use np.random in this environment, but if we did, we would seed it here:
            # self.np_random = np.random.default_rng(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        self.wave_number = 0  # Will be incremented to 1 by _prepare_next_wave

        self.zombies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.wave_cooldown = 0
        self.zombies_to_spawn = []
        self.zombie_spawn_timer = 0

        self._generate_path()
        self._prepare_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0

        self._handle_input(action)

        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_zombies()
        self._update_particles()
        reward += self._update_wave_manager()

        self.steps += 1

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated:
            if self.game_won:
                reward += 100
            elif self.base_health <= 0:
                reward -= 100

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- Cycle Tower (on key press) ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: UI_cycle.wav

        # --- Place Tower (on key press) ---
        if space_held and not self.prev_space_held:
            self._place_tower()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _place_tower(self):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.resources >= spec["cost"]:
            cx, cy = self.cursor_pos
            # Check if tile is empty and not on the path
            is_occupied = any(t['grid_pos'] == [cx, cy] for t in self.towers)
            is_on_path = any(p == (cx, cy) for p in self.path)

            if not is_occupied and not is_on_path:
                self.resources -= spec["cost"]
                px, py = (cx + 0.5) * self.TILE_SIZE, (cy + 0.5) * self.TILE_SIZE
                new_tower = {
                    "spec_idx": self.selected_tower_type,
                    "grid_pos": [cx, cy],
                    "pos": (px, py),
                    "last_fire_step": self.steps,
                    "fire_flash": 0,
                }
                self.towers.append(new_tower)
                self._create_particles(px, py, spec['color'], 20, 2, 15)
                # sfx: place_tower.wav

    def _update_wave_manager(self):
        reward = 0
        # If all zombies are spawned and defeated, start cooldown for the next wave
        if not self.zombies and not self.zombies_to_spawn and self.wave_cooldown <= 0:
            if self.wave_number > 0:  # Don't reward for wave 0 completing
                reward += 1  # Wave complete reward
                # sfx: wave_complete.wav

            if self.wave_number >= self.MAX_WAVES:
                self.game_won = True
            else:
                self.wave_cooldown = self.WAVE_COOLDOWN_STEPS

        if self.wave_cooldown > 0:
            self.wave_cooldown -= 1
            if self.wave_cooldown == 0:
                self._prepare_next_wave()

        # Spawn zombies from the queue
        if self.zombies_to_spawn and self.zombie_spawn_timer <= 0:
            zombie_spec = self.zombies_to_spawn.pop(0)
            start_pos = (self.path[0][0] + 0.5) * self.TILE_SIZE, (self.path[0][1] + 0.5) * self.TILE_SIZE
            self.zombies.append({
                "pos": list(start_pos),
                "path_idx": 0,
                "health": zombie_spec['health'],
                "max_health": zombie_spec['health'],
                "speed": zombie_spec['speed'],
                "slow_timer": 0,
            })
            self.zombie_spawn_timer = zombie_spec['spawn_delay']

        if self.zombie_spawn_timer > 0:
            self.zombie_spawn_timer -= 1

        return reward

    def _prepare_next_wave(self):
        self.wave_number += 1
        if self.wave_number > 1:
            self.resources += self.RESOURCES_PER_WAVE

        num_zombies = 5 + self.wave_number * 2
        base_health = 50 * (1 + (self.wave_number - 1) * 0.15)
        base_speed = 0.8 * (1 + (self.wave_number - 1) * 0.05)
        spawn_delay = max(10, 30 - self.wave_number)

        self.zombies_to_spawn = []
        for _ in range(num_zombies):
            self.zombies_to_spawn.append({
                'health': base_health,
                'speed': base_speed,
                'spawn_delay': spawn_delay
            })

    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['spec_idx']]

            if tower['fire_flash'] > 0:
                tower['fire_flash'] -= 1

            # Handle frost tower area slow effect
            if spec['name'] == 'Frost':
                if self.steps - tower['last_fire_step'] >= spec['fire_rate']:
                    tower['last_fire_step'] = self.steps
                    for z in self.zombies:
                        dist = math.hypot(z['pos'][0] - tower['pos'][0], z['pos'][1] - tower['pos'][1])
                        if dist <= spec['range']:
                            z['slow_timer'] = max(z['slow_timer'], 5)  # Re-apply slow every 5 frames
                continue  # Frost tower doesn't fire projectiles

            # Handle projectile towers
            if self.steps - tower['last_fire_step'] >= spec['fire_rate']:
                target = None
                min_dist = float('inf')
                for z in self.zombies:
                    dist = math.hypot(z['pos'][0] - tower['pos'][0], z['pos'][1] - tower['pos'][1])
                    if dist < spec['range'] and dist < min_dist:
                        min_dist = dist
                        target = z

                if target:
                    tower['last_fire_step'] = self.steps
                    tower['fire_flash'] = 3
                    self.projectiles.append({
                        "pos": list(tower['pos']),
                        "target": target,
                        "spec_idx": tower['spec_idx']
                    })
                    # sfx: fire_gatling.wav or fire_cannon.wav
        return 0

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            spec = self.TOWER_SPECS[p['spec_idx']]
            target = p['target']

            if target not in self.zombies:  # Target already dead
                self.projectiles.remove(p)
                continue

            # Move towards target
            dx = target['pos'][0] - p['pos'][0]
            dy = target['pos'][1] - p['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < spec['proj_speed']:
                # Hit
                target['health'] -= spec['damage']
                self._create_particles(p['pos'][0], p['pos'][1], spec['color'], 10, 1, 10)
                # sfx: hit_impact.wav
                if target['health'] <= 0:
                    self._create_particles(target['pos'][0], target['pos'][1], self.COLOR_ZOMBIE, 30, 3, 20)
                    self.zombies.remove(target)
                    reward += 0.1  # Kill reward
                    # sfx: zombie_death.wav
                self.projectiles.remove(p)
            else:
                p['pos'][0] += (dx / dist) * spec['proj_speed']
                p['pos'][1] += (dy / dist) * spec['proj_speed']
        return reward

    def _update_zombies(self):
        reward = 0
        for z in self.zombies[:]:
            current_speed = z['speed']
            if z['slow_timer'] > 0:
                spec = self.TOWER_SPECS[2]  # Frost tower spec
                current_speed *= spec['slow_factor']
                z['slow_timer'] -= 1

            if z['path_idx'] >= len(self.path) - 1:
                # Reached the base
                self.base_health -= 10
                self._create_particles(z['pos'][0], z['pos'][1], self.COLOR_BASE_DMG, 20, 2, 15)
                self.zombies.remove(z)
                reward -= 0.1  # Penalty for reaching base
                # sfx: base_damage.wav
                continue

            target_node = self.path[z['path_idx'] + 1]
            target_pos = (target_node[0] + 0.5) * self.TILE_SIZE, (target_node[1] + 0.5) * self.TILE_SIZE

            dx = target_pos[0] - z['pos'][0]
            dy = target_pos[1] - z['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < current_speed:
                z['path_idx'] += 1
                z['pos'][0] = target_pos[0]
                z['pos'][1] = target_pos[1]
            else:
                z['pos'][0] += (dx / dist) * current_speed
                z['pos'][1] += (dy / dist) * current_speed
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05  # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, x, y, color, count, speed_scale, life):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 1.5) * speed_scale
            self.particles.append({
                "pos": [x, y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "color": color,
                "life": random.randint(life // 2, life)
            })

    def _check_termination(self):
        if self.base_health <= 0:
            return True
        if self.game_won:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw path
        if len(self.path) > 1:
            for i in range(len(self.path)):
                rect = pygame.Rect(self.path[i][0] * self.TILE_SIZE, self.path[i][1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_PATH, rect)

        # Draw base (end of path)
        base_node = self.path[-1]
        base_rect = pygame.Rect(base_node[0] * self.TILE_SIZE, base_node[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)

        # Draw towers
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['spec_idx']]
            px, py = int(tower['pos'][0]), int(tower['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, px, py, self.TILE_SIZE // 3, spec['color'])
            pygame.gfxdraw.aacircle(self.screen, px, py, self.TILE_SIZE // 3, spec['color'])
            if tower['fire_flash'] > 0:
                alpha = int(255 * (tower['fire_flash'] / 3))
                s = pygame.Surface((self.TILE_SIZE // 2, self.TILE_SIZE // 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (255, 255, 255, alpha), (self.TILE_SIZE // 4, self.TILE_SIZE // 4), self.TILE_SIZE // 4)
                self.screen.blit(s, (px - self.TILE_SIZE // 4, py - self.TILE_SIZE // 4))

        # Draw projectiles
        for p in self.projectiles:
            spec = self.TOWER_SPECS[p['spec_idx']]
            px, py = int(p['pos'][0]), int(p['pos'][1])
            pygame.draw.circle(self.screen, spec['color'], (px, py), 4)

        # Draw zombies
        for z in self.zombies:
            px, py = int(z['pos'][0]), int(z['pos'][1])
            color = self.COLOR_ZOMBIE
            if z['slow_timer'] > 0:
                color = (100, 100, 220)  # Blue tint when slowed
            pygame.draw.rect(self.screen, color, (px - 8, py - 8, 16, 16))
            # Health bar
            health_pct = z['health'] / z['max_health']
            pygame.draw.rect(self.screen, (50, 50, 50), (px - 10, py - 15, 20, 4))
            pygame.draw.rect(self.screen, (50, 200, 50), (px - 10, py - 15, int(20 * health_pct), 4))

        # Draw particles
        for p in self.particles:
            px, py = int(p['pos'][0]), int(p['pos'][1])
            life_pct = p['life'] / 20.0
            size = int(3 * life_pct)
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], (px, py), size)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(cx * self.TILE_SIZE, cy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)

        spec = self.TOWER_SPECS[self.selected_tower_type]
        can_place = self.resources >= spec['cost'] and not any(t['grid_pos'] == [cx, cy] for t in self.towers) and not any(p == (cx, cy) for p in self.path)
        cursor_color = (0, 255, 0) if can_place else (255, 0, 0)

        # Draw transparent fill
        s = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        s.fill((*cursor_color, 50))
        self.screen.blit(s, (cursor_rect.x, cursor_rect.y))
        pygame.draw.rect(self.screen, cursor_color, cursor_rect, 2)

        # Draw tower range preview
        if spec['name'] != 'Frost':
            pygame.gfxdraw.aacircle(self.screen, cursor_rect.centerx, cursor_rect.centery, int(spec['range']), (*cursor_color, 150))
        else:  # Frost tower has an area effect, show as filled
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.circle(s, (*cursor_color, 30), (cursor_rect.centerx, cursor_rect.centery), int(spec['range']))
            self.screen.blit(s, (0,0))

    def _render_text_with_outline(self, font, text, pos, color, outline_color):
        text_surface = font.render(text, True, color)
        outline_surface = font.render(text, True, outline_color)
        self.screen.blit(outline_surface, (pos[0] - 1, pos[1] - 1))
        self.screen.blit(outline_surface, (pos[0] + 1, pos[1] - 1))
        self.screen.blit(outline_surface, (pos[0] - 1, pos[1] + 1))
        self.screen.blit(outline_surface, (pos[0] + 1, pos[1] + 1))
        self.screen.blit(text_surface, pos)

    def _render_ui(self):
        # Top-right UI Panel
        self._render_text_with_outline(self.font_ui, f"Wave: {self.wave_number}/{self.MAX_WAVES}", (480, 10), self.COLOR_TEXT, self.COLOR_TEXT_OUTLINE)
        self._render_text_with_outline(self.font_ui, f"Base HP: {max(0, self.base_health)}", (480, 30), self.COLOR_TEXT, self.COLOR_TEXT_OUTLINE)
        self._render_text_with_outline(self.font_ui, f"Resources: ${self.resources}", (480, 50), self.COLOR_TEXT, self.COLOR_TEXT_OUTLINE)

        # Bottom-left Selected Tower Panel
        spec = self.TOWER_SPECS[self.selected_tower_type]
        self._render_text_with_outline(self.font_title, f"Selected: {spec['name']}", (10, 340), self.COLOR_TEXT, self.COLOR_TEXT_OUTLINE)
        self._render_text_with_outline(self.font_ui, f"Cost: ${spec['cost']}", (10, 370), self.COLOR_TEXT, self.COLOR_TEXT_OUTLINE)

        # Game Over / Win message
        if self._check_termination() and self.steps < self.MAX_STEPS:
            message = "YOU WON!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            self._render_text_with_outline(pygame.font.Font(None, 80), message, (120, 160), color, self.COLOR_TEXT_OUTLINE)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
            "resources": self.resources,
            "zombies_remaining": len(self.zombies) + len(self.zombies_to_spawn)
        }

    def _generate_path(self):
        self.path = []
        self.path.append((0, 4))
        self.path.append((3, 4))
        self.path.append((3, 1))
        self.path.append((8, 1))
        self.path.append((8, 8))
        self.path.append((12, 8))
        self.path.append((12, 4))
        self.path.append((self.GRID_WIDTH - 1, 4))

    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # It will not run in a headless environment (e.g. a server)
    # as it requires a display.
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass # Variable was not set, which is fine.

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0  # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        if terminated or truncated:
            print(f"Game Over. Final Score: {info['score']}. Resetting in 3 seconds...")
            pygame.time.wait(3000)
            obs, info = env.reset()

        clock.tick(30)  # Limit to 30 FPS

    pygame.quit()