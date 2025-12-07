import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium import spaces
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    A turn-based strategy game where the player places animal clones to collect water.

    The goal is to maximize the water collected within a fixed number of turns,
    while avoiding detection by rival animals. The game is presented on a grid-based
    desert map.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]` (Movement): 0=none, 1=up, 2=down, 3=left, 4=right. Moves the placement cursor.
    - `action[1]` (Space): 0=released, 1=held. Places the selected animal card clone.
    - `action[2]` (Shift): 0=released, 1=held. Cycles through the available animal cards.

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - +0.1 per unit of water collected.
    - -1.0 per clone detected and removed by a rival.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A turn-based strategy game where you place animal clones to collect water from a desert map, "
        "while avoiding detection by rival animals."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press Shift to cycle through animal cards "
        "and press Space to place a clone on the selected tile."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 10, 5
    CELL_SIZE = 64
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (194, 178, 128)  # Desert Sand
    COLOR_GRID = (161, 145, 112)
    COLOR_WATER = (70, 130, 180)
    COLOR_PLAYER = (50, 205, 50)
    COLOR_RIVAL = (220, 20, 60)
    COLOR_CURSOR = (255, 215, 0, 150) # Gold with alpha
    COLOR_TEXT = (47, 79, 79)
    COLOR_SANDSTORM = (210, 180, 140, 100)

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = (0, 0)
        self.player_clones = []
        self.rival_animals = []
        self.water_sources = []
        self.particles = []
        self.visual_effects = []

        self.player_cards = [
            {"name": "Scorpion", "range": 2, "collect": 1, "color": (60, 179, 113)}, # SeaGreen
            {"name": "Lizard", "range": 3, "collect": 2, "color": (154, 205, 50)}, # YellowGreen
            {"name": "Fennec Fox", "range": 4, "collect": 3, "color": (34, 139, 34)}, # ForestGreen
        ]
        self.selected_card_idx = 0
        self.last_space_held = False
        self.last_shift_held = False

        self.sandstorm_active = False
        self.sandstorm_opacity = 0
        self.total_initial_water = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = (self.GRID_COLS // 2, self.GRID_ROWS // 2)

        self.player_clones = []
        self.particles = []
        self.visual_effects = []

        self.selected_card_idx = 0
        self.last_space_held = False
        self.last_shift_held = False

        self.sandstorm_active = False
        self.sandstorm_opacity = 0

        # Procedurally generate the level
        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # Generate Water Sources
        self.water_sources = []
        num_sources = self.np_random.integers(2, 5)
        self.total_initial_water = 0
        for _ in range(num_sources):
            pos = (self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS))
            # Ensure no overlap
            while any(w['pos'] == pos for w in self.water_sources):
                pos = (self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS))
            amount = self.np_random.integers(50, 101)
            self.water_sources.append({"pos": pos, "amount": amount, "initial_amount": amount})
            self.total_initial_water += amount

        # Generate Rival Animals
        self.rival_animals = []
        num_rivals = self.np_random.integers(1, 4)
        for _ in range(num_rivals):
            path_len = self.np_random.integers(2, 5)
            patrol_path = []
            for i in range(path_len):
                pt = (self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS))
                # Ensure no overlap with water or other path points
                while any(w['pos'] == pt for w in self.water_sources) or pt in patrol_path:
                    pt = (self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS))
                patrol_path.append(pt)

            self.rival_animals.append({
                "pos": patrol_path[0],
                "patrol_path": patrol_path,
                "patrol_index": 0,
                "speed": 1.0,
                "move_cooldown": 0
            })

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- 1. Handle Player Input ---
        self._handle_input(movement, space_held, shift_held)

        # --- 2. Advance Game Turn ---
        self.steps += 1

        # Update Clones
        reward += self._update_clones()

        # Update Rivals
        reward += self._update_rivals()

        # Update Environment
        self._update_environment()

        # Update Particles and Effects
        self._update_particles()
        self._update_visual_effects()

        # --- 3. Check Termination ---
        terminated = self.steps >= self.MAX_STEPS
        truncated = False
        if terminated:
            self.game_over = True

        # --- 4. Assertions ---
        assert self.score <= self.total_initial_water, "Score exceeds total possible water"
        assert len(self.player_clones) <= len(self.player_cards), "More clones than cards"
        for rival in self.rival_animals:
            rx, ry = rival['pos']
            assert 0 <= rx < self.GRID_COLS and 0 <= ry < self.GRID_ROWS, "Rival out of bounds"

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        cx, cy = self.cursor_pos
        if movement == 1: cy = max(0, cy - 1)
        elif movement == 2: cy = min(self.GRID_ROWS - 1, cy + 1)
        elif movement == 3: cx = max(0, cx - 1)
        elif movement == 4: cx = min(self.GRID_COLS - 1, cx + 1)
        self.cursor_pos = (cx, cy)

        if shift_held and not self.last_shift_held:
            # SFX: UI_Cycle.wav
            self.selected_card_idx = (self.selected_card_idx + 1) % len(self.player_cards)

        if space_held and not self.last_space_held:
            self._place_clone()

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _place_clone(self):
        is_occupied = any(c['pos'] == self.cursor_pos for c in self.player_clones) or \
                      any(r['pos'] == self.cursor_pos for r in self.rival_animals)

        card = self.player_cards[self.selected_card_idx]
        is_card_already_placed = any(c['card']['name'] == card['name'] for c in self.player_clones)

        if not is_occupied and not is_card_already_placed:
            self.player_clones.append({
                "pos": self.cursor_pos,
                "card": card,
                "target_water": None,
                "path": []
            })
            # SFX: Clone_Place.wav
            self._create_visual_effect(self.cursor_pos, card['color'], 'burst')

    def _update_clones(self):
        reward = 0
        for clone in self.player_clones:
            # Find nearest water source if needed
            if clone['target_water'] is None or clone['target_water']['amount'] <= 0:
                active_sources = [w for w in self.water_sources if w['amount'] > 0]
                if not active_sources:
                    clone['path'] = []
                    continue

                distances = [self._manhattan_distance(clone['pos'], w['pos']) for w in active_sources]
                closest_source = active_sources[np.argmin(distances)]
                clone['target_water'] = closest_source
                clone['path'] = self._find_path(clone['pos'], closest_source['pos'])

            # Move along path
            if clone['path']:
                clone['pos'] = clone['path'].pop(0)

            # Collect water
            if clone['pos'] == clone['target_water']['pos']:
                card = clone['card']
                amount_to_collect = min(card['collect'], clone['target_water']['amount'])
                if amount_to_collect > 0:
                    self.score += amount_to_collect
                    clone['target_water']['amount'] -= amount_to_collect
                    reward += amount_to_collect * 0.1
                    # SFX: Water_Collect.wav
                    self._create_particles(clone['pos'], self.COLOR_WATER, 10, 'to_ui')

                # Find new target if source is depleted
                if clone['target_water']['amount'] <= 0:
                    clone['target_water'] = None
        return reward

    def _update_rivals(self):
        reward = 0
        speed_increase = 0.1 * (self.steps // 500)

        for rival in self.rival_animals:
            rival['speed'] = 1.0 + speed_increase
            rival['move_cooldown'] -= 1
            if rival['move_cooldown'] <= 0:
                rival['patrol_index'] = (rival['patrol_index'] + 1) % len(rival['patrol_path'])
                rival['pos'] = rival['patrol_path'][rival['patrol_index']]
                rival['move_cooldown'] = 10 / rival['speed'] # Cooldown is inverse of speed

            # Check for detection
            detection_radius = 1 if self.sandstorm_active else 2
            clones_to_remove = []
            for clone in self.player_clones:
                if self._manhattan_distance(rival['pos'], clone['pos']) <= detection_radius:
                    clones_to_remove.append(clone)

            for clone_to_remove in clones_to_remove:
                if clone_to_remove in self.player_clones:
                    self.player_clones.remove(clone_to_remove)
                    reward -= 1.0
                    # SFX: Detection_Alert.wav
                    self._create_visual_effect(rival['pos'], self.COLOR_RIVAL, 'pulse')

        return reward

    def _update_environment(self):
        # Sandstorm logic
        if not self.sandstorm_active and self.np_random.random() < 0.005:
            self.sandstorm_active = True
            # SFX: Sandstorm_Start.wav
        elif self.sandstorm_active and self.np_random.random() < 0.01:
            self.sandstorm_active = False
            # SFX: Sandstorm_End.wav

        if self.sandstorm_active:
            self.sandstorm_opacity = min(100, self.sandstorm_opacity + 5)
        else:
            self.sandstorm_opacity = max(0, self.sandstorm_opacity - 5)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        self._render_water_sources()
        self._render_rivals()
        self._render_clones()
        self._render_cursor()
        self._render_particles()
        self._render_visual_effects()
        self._render_sandstorm()

    def _render_grid(self):
        for x in range(self.GRID_COLS + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_Y_OFFSET), (px, self.GRID_Y_OFFSET + self.GRID_HEIGHT))
        for y in range(self.GRID_ROWS + 1):
            py = self.GRID_Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.GRID_WIDTH, py))

    def _render_water_sources(self):
        for source in self.water_sources:
            if source['amount'] > 0:
                cx, cy = self._grid_to_pixel(source['pos'])
                radius = int((self.CELL_SIZE / 2 - 5) * (source['amount'] / source['initial_amount']))
                radius = max(5, radius)
                pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, self.COLOR_WATER)
                pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, self.COLOR_WATER)

    def _render_clones(self):
        for clone in self.player_clones:
            cx, cy = self._grid_to_pixel(clone['pos'])
            color = clone['card']['color']
            self._draw_glowing_shape(self.screen, 'triangle', (cx, cy), self.CELL_SIZE // 4, color)

    def _render_rivals(self):
        for rival in self.rival_animals:
            cx, cy = self._grid_to_pixel(rival['pos'])
            self._draw_glowing_shape(self.screen, 'square', (cx, cy), self.CELL_SIZE // 3, self.COLOR_RIVAL)

    def _render_cursor(self):
        cx, cy = self._grid_to_pixel(self.cursor_pos)
        rect = pygame.Rect(cx - self.CELL_SIZE // 2, cy - self.CELL_SIZE // 2, self.CELL_SIZE, self.CELL_SIZE)

        # Create a temporary surface for transparency
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_CURSOR, s.get_rect(), border_radius=5)
        self.screen.blit(s, rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR[:3], rect, 2, border_radius=5)

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['size']))

    def _render_visual_effects(self):
        for fx in self.visual_effects:
            if fx['type'] == 'pulse':
                alpha = int(255 * (fx['life'] / fx['max_life']))
                color = (*fx['color'], alpha)
                radius = int(fx['size'] * (1 - (fx['life'] / fx['max_life'])))
                cx, cy = self._grid_to_pixel(fx['pos'])
                pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, color)
            elif fx['type'] == 'burst':
                # Handled by particle creation
                pass

    def _render_sandstorm(self):
        if self.sandstorm_opacity > 0:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            color = (*self.COLOR_SANDSTORM[:3], self.sandstorm_opacity)
            overlay.fill(color)
            self.screen.blit(overlay, (0, 0))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"💧 {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Turn
        turn_text = self.font_large.render(f"Turn: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(turn_text, (self.SCREEN_WIDTH - turn_text.get_width() - 10, 10))

        # Selected Card
        card_y = self.SCREEN_HEIGHT - 35
        card_text_label = self.font_small.render("Selected Card:", True, self.COLOR_TEXT)
        self.screen.blit(card_text_label, (10, card_y))

        selected_card = self.player_cards[self.selected_card_idx]
        card_info = f"{selected_card['name']} (Move: {selected_card['range']}, Collect: {selected_card['collect']})"
        card_text = self.font_small.render(card_info, True, selected_card['color'])
        self.screen.blit(card_text, (card_text_label.get_width() + 20, card_y))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Helper Functions ---
    def _grid_to_pixel(self, grid_pos):
        x, y = grid_pos
        px = x * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(px), int(py)

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _find_path(self, start, end):
        # Simple A* pathfinding (manhattan distance heuristic)
        open_set = {start}
        came_from = {}
        g_score = { (x,y): float('inf') for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS) }
        g_score[start] = 0
        f_score = { (x,y): float('inf') for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS) }
        f_score[start] = self._manhattan_distance(start, end)

        while open_set:
            current = min(open_set, key=lambda pos: f_score.get(pos, float('inf')))

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            open_set.remove(current)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.GRID_COLS and 0 <= neighbor[1] < self.GRID_ROWS:
                    tentative_g_score = g_score.get(current, float('inf')) + 1
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._manhattan_distance(neighbor, end)
                        if neighbor not in open_set:
                            open_set.add(neighbor)
        return []

    def _create_particles(self, grid_pos, color, count, behavior='burst'):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(count):
            if behavior == 'burst':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 3)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life = self.np_random.integers(10, 20)
            elif behavior == 'to_ui':
                target_pos = np.array([50, 25])
                start_pos = np.array([px, py])
                direction = target_pos - start_pos
                vel = direction / np.linalg.norm(direction) * self.np_random.uniform(3, 5)
                life = 30

            self.particles.append({
                'pos': [px, py], 'vel': vel, 'size': self.np_random.uniform(2, 4),
                'life': life, 'max_life': life, 'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] *= 0.95
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _create_visual_effect(self, grid_pos, color, fx_type):
        if fx_type == 'pulse':
            self.visual_effects.append({
                'pos': grid_pos, 'color': color, 'type': 'pulse',
                'life': 20, 'max_life': 20, 'size': self.CELL_SIZE
            })
        elif fx_type == 'burst':
            self._create_particles(grid_pos, color, 20, 'burst')

    def _update_visual_effects(self):
        for fx in self.visual_effects:
            fx['life'] -= 1
        self.visual_effects = [fx for fx in self.visual_effects if fx['life'] > 0]

    def _draw_glowing_shape(self, surface, shape, pos, size, color):
        cx, cy = pos
        for i in range(4, 0, -1):
            alpha = 80 - i * 15
            glow_color = (*color, alpha)
            glow_size = size + i * 2

            temp_surf = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)

            if shape == 'square':
                pygame.draw.rect(temp_surf, glow_color, (0, 0, glow_size, glow_size), border_radius=3)
            elif shape == 'triangle':
                points = [ (glow_size/2, 0), (glow_size, glow_size), (0, glow_size) ]
                pygame.draw.polygon(temp_surf, glow_color, points)

            surface.blit(temp_surf, (cx - glow_size/2, cy - glow_size/2))

        if shape == 'square':
            pygame.draw.rect(surface, color, (cx - size/2, cy - size/2, size, size), border_radius=3)
        elif shape == 'triangle':
            points = [ (cx, cy - size/2), (cx + size/2, cy + size/2), (cx - size/2, cy + size/2) ]
            pygame.draw.polygon(surface, color, points)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()

    # Set up display for manual play
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Desert Cloning Strategy Game")
    clock = pygame.time.Clock()

    running = True
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Manual Control Mapping ---
        keys = pygame.key.get_pressed()
        action.fill(0) # Reset actions
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4

        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()
            pygame.time.wait(2000)

        clock.tick(10) # Control game speed for manual play

    env.close()