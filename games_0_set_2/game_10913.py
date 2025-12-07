import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:12:24.692630
# Source Brief: brief_00913.md
# Brief Index: 913
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    An action-puzzle Gymnasium environment where the player escapes a procedurally
    generated petrified city. The player teleports between buildings and reverses
    time to freeze creatures and reveal new paths to reach the exit.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Escape a petrified city by teleporting between buildings. "
        "Reverse time to freeze creatures and reveal hidden paths to the exit."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to teleport to the nearest building in that direction. "
        "Press space to reverse time."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    GRID_SIZE = 30
    TILE_ISO_WIDTH = 32
    TILE_ISO_HEIGHT = 16
    BUILDING_HEIGHT = 24

    # --- Colors ---
    COLOR_BG = (15, 20, 35)
    COLOR_BUILDING = (70, 80, 90)
    COLOR_BUILDING_REVERSED = (90, 100, 120)
    COLOR_BUILDING_REVERSED_ACTIVE = (120, 150, 180)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 255, 255, 60)
    COLOR_CREATURE = (255, 50, 50)
    COLOR_EXIT_BEACON = (255, 255, 100)
    COLOR_TELEPORT_TARGET = (200, 255, 200)
    OVERLAY_NORMAL = (50, 50, 200, 30)
    OVERLAY_REVERSED = (220, 160, 50, 30)
    COLOR_PLANT = (50, 200, 80)
    COLOR_TEXT = (240, 240, 240)

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
        self.font_large = pygame.font.Font(None, 36)

        # Game state variables are initialized in reset()
        self.buildings = []
        self.creatures = []
        self.particles = []
        self.player_building_idx = 0
        self.exit_building_idx = 0
        self.time_state = 'normal'
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.num_creatures = 1
        self.camera_pos = pygame.Vector2(0, 0)
        self.camera_target = pygame.Vector2(0, 0)
        self.teleport_targets = {}
        self.plant_growth = 0.0

        # The reset call is necessary to initialize the game state for the first time
        # self.reset() # This can be removed if the user is expected to call it.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._generate_city()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_state = 'normal'
        self.last_space_held = False
        self.particles = []
        self.num_creatures = 1

        player_building = self.buildings[self.player_building_idx]
        self.camera_pos.update(self._grid_to_screen(player_building['grid_pos']))
        self.camera_target.update(self.camera_pos)

        self._update_teleport_targets()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1

        dist_to_exit_before = self._get_dist_to_exit()
        dist_to_nearest_creature_before = self._get_dist_to_nearest_creature()

        # --- Action Handling ---
        # 1. Time Reversal (Space Bar Press)
        time_reversed_this_step = False
        if space_held and not self.last_space_held:
            # Sound: TimeWarp.wav
            time_reversed_this_step = True
            self.time_state = 'reversed' if self.time_state == 'normal' else 'normal'
            self._create_time_warp_particles()
            self._update_teleport_targets() # Targets may change
        self.last_space_held = space_held

        # 2. Teleportation (Movement)
        if movement != 0 and movement in self.teleport_targets:
            # Sound: Teleport.wav
            old_pos_screen = self._grid_to_screen(self.buildings[self.player_building_idx]['grid_pos'])
            self._create_teleport_particles(old_pos_screen, 'out')

            self.player_building_idx = self.teleport_targets[movement]

            new_pos_screen = self._grid_to_screen(self.buildings[self.player_building_idx]['grid_pos'])
            self._create_teleport_particles(new_pos_screen, 'in')

            self.camera_target.update(new_pos_screen)
            self._update_teleport_targets()

        # --- Game Logic Update ---
        if self.time_state == 'normal':
            self.plant_growth = max(0.0, self.plant_growth - 0.1)
            for creature in self.creatures:
                creature['pos_on_building'] += creature['dir'] * 0.05
                if not (0 <= creature['pos_on_building'] <= 1):
                    creature['dir'] *= -1
                    creature['pos_on_building'] = np.clip(creature['pos_on_building'], 0, 1)
        else: # Reversed time
            self.plant_growth = min(1.0, self.plant_growth + 0.1)

        self._update_particles()
        self.camera_pos.move_towards_ip(self.camera_target, 20.0)

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 500 == 0:
            if self.num_creatures < len(self.buildings) // 4:
                self.num_creatures += 1
                self._add_creature()

        # --- Termination Check & Reward ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if self.buildings[self.player_building_idx]['is_exit']:
            reward += 100
            terminated = True
        if self._check_collision():
            reward -= 100
            terminated = True

        if terminated or truncated:
            self.game_over = True

        dist_to_exit_after = self._get_dist_to_exit()
        reward += (dist_to_exit_before - dist_to_exit_after) * 0.1

        dist_to_nearest_creature_after = self._get_dist_to_nearest_creature()
        if dist_to_nearest_creature_after < dist_to_nearest_creature_before:
            reward -= 5

        if time_reversed_this_step and dist_to_nearest_creature_before < 4:
            reward += 5

        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _generate_city(self):
        while True:
            self.buildings = []
            grid = {}
            start_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
            grid[start_pos] = 0
            self.buildings.append({'grid_pos': start_pos, 'type': 'normal', 'is_exit': False})
            active_indices = [0]

            while active_indices:
                idx = random.choice(active_indices)
                base_pos = self.buildings[idx]['grid_pos']
                found_neighbor = False
                directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                random.shuffle(directions)

                for dx, dy in directions:
                    new_pos = (base_pos[0] + dx, base_pos[1] + dy)
                    if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE and new_pos not in grid:
                        b_type = 'reversed_only' if random.random() < 0.2 else 'normal'
                        new_idx = len(self.buildings)
                        grid[new_pos] = new_idx
                        self.buildings.append({'grid_pos': new_pos, 'type': b_type, 'is_exit': False})
                        active_indices.append(new_idx)
                        found_neighbor = True
                        break

                if not found_neighbor:
                    active_indices.remove(idx)

            self.player_building_idx = grid[start_pos]
            farthest_dist = -1
            for i, b in enumerate(self.buildings):
                dist = self._manhattan_distance(start_pos, b['grid_pos'])
                if dist > farthest_dist:
                    farthest_dist = dist
                    self.exit_building_idx = i

            self.buildings[self.exit_building_idx]['is_exit'] = True

            if self._is_path_to_exit_possible():
                break

        self.creatures = []
        for _ in range(self.num_creatures):
            self._add_creature()

    def _is_path_to_exit_possible(self):
        q = deque([self.player_building_idx])
        visited = {self.player_building_idx}
        building_map = {b['grid_pos']: i for i, b in enumerate(self.buildings)}

        while q:
            current_idx = q.popleft()
            if current_idx == self.exit_building_idx:
                return True
            current_pos = self.buildings[current_idx]['grid_pos']
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                neighbor_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if neighbor_pos in building_map:
                    neighbor_idx = building_map[neighbor_pos]
                    if neighbor_idx not in visited:
                        visited.add(neighbor_idx)
                        q.append(neighbor_idx)
        return False

    def _add_creature(self):
        possible_spawns = [i for i, b in enumerate(self.buildings) if i != self.player_building_idx and i != self.exit_building_idx and i not in [c['building_idx'] for c in self.creatures]]
        if not possible_spawns: return

        creature_building_idx = random.choice(possible_spawns)
        self.creatures.append({
            'building_idx': creature_building_idx,
            'pos_on_building': random.random(),
            'dir': random.choice([-1, 1])
        })

    def _update_teleport_targets(self):
        self.teleport_targets = {}
        player_pos = self.buildings[self.player_building_idx]['grid_pos']
        potential_targets = {}

        for i, building in enumerate(self.buildings):
            if i == self.player_building_idx: continue

            is_traversable = building['type'] == 'normal' or (building['type'] == 'reversed_only' and self.time_state == 'reversed')
            if not is_traversable: continue

            target_pos = building['grid_pos']
            dx, dy = target_pos[0] - player_pos[0], target_pos[1] - player_pos[1]
            dist_sq = dx*dx + dy*dy

            direction = 0
            if dx == 0 and dy < 0: direction = 1 # Up
            elif dx == 0 and dy > 0: direction = 2 # Down
            elif dy == 0 and dx < 0: direction = 3 # Left
            elif dy == 0 and dx > 0: direction = 4 # Right

            if direction != 0:
                if direction not in potential_targets or dist_sq < potential_targets[direction][0]:
                    potential_targets[direction] = (dist_sq, i)

        for direction, (dist, b_idx) in potential_targets.items():
            self.teleport_targets[direction] = b_idx

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS: return True
        if self.buildings[self.player_building_idx]['is_exit']: return True
        if self._check_collision(): return True
        return False

    def _check_collision(self):
        if self.time_state == 'normal':
            for creature in self.creatures:
                if creature['building_idx'] == self.player_building_idx:
                    # Sound: PlayerHit.wav
                    return True
        return False

    def _get_dist_to_exit(self):
        player_pos = self.buildings[self.player_building_idx]['grid_pos']
        exit_pos = self.buildings[self.exit_building_idx]['grid_pos']
        return self._manhattan_distance(player_pos, exit_pos)

    def _get_dist_to_nearest_creature(self):
        if not self.creatures: return float('inf')
        player_pos = self.buildings[self.player_building_idx]['grid_pos']
        return min(self._manhattan_distance(player_pos, self.buildings[c['building_idx']]['grid_pos']) for c in self.creatures)

    @staticmethod
    def _manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _grid_to_screen(self, grid_pos):
        gx, gy = grid_pos
        screen_x = self.SCREEN_WIDTH / 2 + (gx - gy) * self.TILE_ISO_WIDTH / 2
        screen_y = self.SCREEN_HEIGHT / 2 + (gx + gy) * self.TILE_ISO_HEIGHT / 2
        return pygame.Vector2(screen_x, screen_y)

    def _render_all(self):
        offset = self.camera_pos - pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.screen.fill(self.COLOR_BG)

        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        color = self.OVERLAY_REVERSED if self.time_state == 'reversed' else self.OVERLAY_NORMAL
        overlay.fill(color)
        self.screen.blit(overlay, (0, 0))

        render_queue = sorted(self.buildings, key=lambda b: b['grid_pos'][0] + b['grid_pos'][1])
        all_entities = render_queue + self.creatures + ['player']

        # This is a simplified z-sorting. A more robust solution would interleave entities.
        for building in render_queue: self._render_building(building, offset)
        for creature in self.creatures: self._render_creature(creature, offset)
        self._render_player(offset)

        for p in self.particles:
            pos = (p['pos'][0] - offset.x, p['pos'][1] - offset.y)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(p['radius']), p['color'])

        self._render_ui()

    def _render_building(self, building, offset):
        center_pos = self._grid_to_screen(building['grid_pos']) - offset
        is_target = building in [self.buildings[idx] for idx in self.teleport_targets.values()]

        if building['type'] == 'reversed_only':
            base_color = self.COLOR_BUILDING_REVERSED_ACTIVE if self.time_state == 'reversed' else self.COLOR_BUILDING_REVERSED
        else: base_color = self.COLOR_BUILDING

        top_color = tuple(np.clip(np.array(base_color) * 1.2, 0, 255))
        if is_target: top_color = self.COLOR_TELEPORT_TARGET
        side_color = tuple(np.clip(np.array(base_color) * 0.8, 0, 255))

        w, h = self.TILE_ISO_WIDTH, self.TILE_ISO_HEIGHT
        x, y, z = center_pos.x, center_pos.y, self.BUILDING_HEIGHT
        points = [(x, y - h / 2), (x + w / 2, y), (x, y + h / 2), (x - w / 2, y)]

        pygame.gfxdraw.aapolygon(self.screen, points, top_color); pygame.gfxdraw.filled_polygon(self.screen, points, top_color)
        side1 = [points[2], points[3], (points[3][0], points[3][1] + z), (points[2][0], points[2][1] + z)]
        side2 = [points[1], points[2], (points[2][0], points[2][1] + z), (points[1][0], points[1][1] + z)]
        pygame.gfxdraw.filled_polygon(self.screen, side1, side_color); pygame.gfxdraw.filled_polygon(self.screen, side2, side_color)
        pygame.gfxdraw.aapolygon(self.screen, side1, side_color); pygame.gfxdraw.aapolygon(self.screen, side2, side_color)

        if building['type'] == 'reversed_only' and self.plant_growth > 0:
            for i in range(3): pygame.draw.line(self.screen, self.COLOR_PLANT, (x + (i - 1) * 5, y + h/4), (x + (i - 1) * 5, y + h/4 - self.plant_growth * 15), 2)

        if building['is_exit']:
            pulse = (math.sin(self.steps * 0.1) + 1) / 2
            start_pos, end_pos = (int(x), int(y - 10)), (int(x), int(y - (50 + pulse * 20)))
            pygame.draw.line(self.screen, self.COLOR_EXIT_BEACON, start_pos, end_pos, 2)
            pygame.gfxdraw.filled_circle(self.screen, end_pos[0], end_pos[1], int(5 + pulse * 3), self.COLOR_EXIT_BEACON)

    def _render_player(self, offset):
        pos = self._grid_to_screen(self.buildings[self.player_building_idx]['grid_pos']) - offset
        y_pos = pos.y - 8 + math.sin(self.steps * 0.2) * 3
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(y_pos), 12, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(y_pos), 12, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(y_pos), 6, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(y_pos), 6, self.COLOR_PLAYER)

    def _render_creature(self, creature, offset):
        center_pos = self._grid_to_screen(self.buildings[creature['building_idx']]['grid_pos'])
        w, h = self.TILE_ISO_WIDTH, self.TILE_ISO_HEIGHT
        start_x, start_y = center_pos.x - w / 4, center_pos.y + h / 4
        end_x, end_y = center_pos.x + w / 4, center_pos.y - h / 4

        pos_x = start_x + (end_x - start_x) * creature['pos_on_building']
        pos_y = start_y + (end_y - start_y) * creature['pos_on_building']
        pos = pygame.Vector2(pos_x, pos_y) - offset

        color = self.COLOR_CREATURE if self.time_state == 'normal' else tuple(np.clip(np.array(self.COLOR_CREATURE) * 0.5 + np.array(self.COLOR_BUILDING) * 0.5, 0, 255))
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y - 5), 5, color)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y - 5), 5, color)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))
        time_color = self.OVERLAY_REVERSED[:3] if self.time_state == 'reversed' else self.OVERLAY_NORMAL[:3]
        time_text = self.font_large.render(f"TIME: {self.time_state.upper()}", True, time_color)
        self.screen.blit(time_text, (self.SCREEN_WIDTH / 2 - time_text.get_width() / 2, 10))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']; p['life'] -= 1; p['radius'] *= 0.95

    def _create_teleport_particles(self, pos, direction='out'):
        # Sound: TeleportEffect.wav
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            start_pos = pos
            if direction == 'in': vel *= -1; start_pos += vel * 8
            self.particles.append({'pos': pygame.Vector2(start_pos), 'vel': vel, 'radius': random.uniform(2, 5), 'color': self.COLOR_PLAYER, 'life': 20})

    def _create_time_warp_particles(self):
        # Sound: TimeWarpEffect.wav
        color = self.OVERLAY_REVERSED if self.time_state == 'reversed' else self.OVERLAY_NORMAL
        for _ in range(50): self.particles.append({'pos': pygame.Vector2(random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)), 'vel': pygame.Vector2(random.uniform(-1, 1), 0), 'radius': random.uniform(5, 15), 'color': (color[0], color[1], color[2], random.randint(50, 150)), 'life': 40})

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and visualization
    # It will not be run by the evaluation server
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a visible driver
    env = GameEnv()
    obs, info = env.reset()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Petrified City")
    clock = pygame.time.Clock()
    running = True

    while running:
        movement, space_held, shift_held = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        # shift_held is not used in the game logic, but we can map it
        # if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        obs, reward, terminated, truncated, info = env.step([movement, space_held, shift_held])

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Reset the environment to play again
            obs, info = env.reset()

        clock.tick(30)

    env.close()