import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:49:50.838651
# Source Brief: brief_02481.md
# Brief Index: 2481
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import colorsys

class GameEnv(gym.Env):
    """
    GameEnv: Conveyor Belt Chaos
    The player's goal is to swap items on three horizontally moving conveyor belts
    to create matches of three or more of the same color.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Cursor Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - action[1]: Select/Swap (0: released, 1: pressed)
    - action[2]: Cancel Selection (0: released, 1: pressed)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +1 for each item in a match.
    - +5 for each match group formed in a chain reaction (2nd wave onwards).
    - +100 for winning the game (reaching the score limit).
    - -10 for losing the game (running out of time).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Swap items on three moving conveyor belts to create matches of three or more of the same color. "
        "Race against the clock to reach the target score!"
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to select an item, move to another, "
        "and press space again to swap. Press shift to cancel a selection."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 90
    
    NUM_BELTS = 3
    ITEM_SIZE = 32
    ITEM_SPACING = 8
    BELT_HEIGHT = 50
    BELT_SPEED = 40.0  # pixels per second

    WIN_SCORE = 500

    # --- COLORS ---
    COLOR_BG = (15, 20, 35)
    COLOR_BELT = (35, 45, 65)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_SELECTED = (0, 255, 255)

    ITEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
        (255, 150, 80),  # Orange
        (150, 80, 255),  # Purple
        (200, 200, 200), # White
        (255, 180, 220), # Pink
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gym Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_timer = pygame.font.SysFont("Consolas", 30, bold=True)

        # --- Game State Initialization ---
        self.belts = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.cursor_pos = (0, 0)
        self.selected_item_coords = None
        self.last_action_states = [0, 0] # space, shift
        self.item_id_counter = 0

        # --- Calculated Geometry ---
        self.BELT_Y_POS = [
            self.SCREEN_HEIGHT // 2 - self.BELT_HEIGHT - 20,
            self.SCREEN_HEIGHT // 2,
            self.SCREEN_HEIGHT // 2 + self.BELT_HEIGHT + 20,
        ]
        self.ITEMS_PER_BELT = self.SCREEN_WIDTH // (self.ITEM_SIZE + self.ITEM_SPACING) + 4

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_remaining = self.GAME_DURATION_SECONDS
        self.game_over = False
        
        self.cursor_pos = (1, self.ITEMS_PER_BELT // 2) # Start in middle
        self.selected_item_coords = None
        self.last_action_states = [0, 0]
        
        self.particles.clear()
        self.item_id_counter = 0
        
        # --- Initialize Belts with Items (No initial matches) ---
        self.belts = []
        for i in range(self.NUM_BELTS):
            belt = []
            for j in range(self.ITEMS_PER_BELT):
                belt.append(self._create_item(j))
            self.belts.append(belt)

        # Ensure no starting matches
        while self._find_and_fix_initial_matches():
            pass

        for i in range(self.NUM_BELTS):
            for j, item in enumerate(self.belts[i]):
                item['x'] = self.SCREEN_WIDTH + j * (self.ITEM_SIZE + self.ITEM_SPACING)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        dt = self.clock.tick(self.FPS) / 1000.0
        self.steps += 1
        self.time_remaining = max(0, self.time_remaining - dt)

        reward = self._handle_input_and_actions(action)
        self._update_world(dt)
        reward += self._process_matches()
        self._update_particles(dt)

        terminated = self.score >= self.WIN_SCORE or self.time_remaining <= 0
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100  # Win bonus
            else:
                reward += -10  # Loss penalty
        
        truncated = False # This environment does not have a step limit
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input_and_actions(self, action):
        movement, space_val, shift_val = action[0], action[1], action[2]
        
        space_press = space_val == 1 and self.last_action_states[0] == 0
        shift_press = shift_val == 1 and self.last_action_states[1] == 0
        self.last_action_states = [space_val, shift_val]

        # --- Cursor Movement ---
        belt_idx, item_idx = self.cursor_pos
        if movement == 1: belt_idx = max(0, belt_idx - 1) # Up
        elif movement == 2: belt_idx = min(self.NUM_BELTS - 1, belt_idx + 1) # Down
        
        # Clamp item index to current belt length
        num_items_on_belt = len(self.belts[belt_idx])
        item_idx = min(item_idx, num_items_on_belt - 1) if num_items_on_belt > 0 else 0

        if movement == 3: item_idx = max(0, item_idx - 1) # Left
        elif movement == 4: item_idx = min(num_items_on_belt - 1, item_idx + 1) if num_items_on_belt > 0 else 0 # Right
        
        self.cursor_pos = (belt_idx, item_idx)

        # --- Actions ---
        if shift_press and self.selected_item_coords:
            self.selected_item_coords = None

        if space_press:
            if not self.selected_item_coords:
                if len(self.belts[self.cursor_pos[0]]) > self.cursor_pos[1]:
                    self.selected_item_coords = self.cursor_pos
            else:
                if self.cursor_pos == self.selected_item_coords:
                    self.selected_item_coords = None # Deselect
                else:
                    # Perform swap
                    sel_belt_idx, sel_item_idx = self.selected_item_coords
                    cur_belt_idx, cur_item_idx = self.cursor_pos

                    if sel_item_idx < len(self.belts[sel_belt_idx]) and cur_item_idx < len(self.belts[cur_belt_idx]):
                        item1 = self.belts[sel_belt_idx][sel_item_idx]
                        item2 = self.belts[cur_belt_idx][cur_item_idx]
                        
                        # Swap color and ID for logic, keep positions
                        item1['color_idx'], item2['color_idx'] = item2['color_idx'], item1['color_idx']
                    
                    self.selected_item_coords = None
        return 0

    def _update_world(self, dt):
        # --- Move Items on Belts ---
        for belt in self.belts:
            for item in belt:
                item['x'] -= self.BELT_SPEED * dt
        
        # --- Spawn/Despawn Items ---
        for i in range(self.NUM_BELTS):
            # Remove items that have moved off-screen
            self.belts[i] = [item for item in self.belts[i] if item['x'] > -self.ITEM_SIZE]
            
            # Add new items to the right if needed
            while len(self.belts[i]) < self.ITEMS_PER_BELT:
                last_item_x = self.SCREEN_WIDTH
                if self.belts[i]:
                    # Find the rightmost item to position the new one correctly
                    rightmost_item = max(self.belts[i], key=lambda it: it['x'])
                    last_item_x = rightmost_item['x']

                new_item = self._create_item(len(self.belts[i]))
                new_item['x'] = last_item_x + self.ITEM_SIZE + self.ITEM_SPACING
                self.belts[i].append(new_item)

    def _process_matches(self):
        total_reward = 0
        chain_level = 0
        while True:
            all_matches = self._find_all_matches()
            if not any(all_matches.values()):
                break
            
            chain_level += 1
            items_to_remove_by_id = set()

            for belt_idx, belt_matches in all_matches.items():
                if not belt_matches: continue

                for match_group in belt_matches:
                    num_in_match = len(match_group)
                    self.score += 10 * num_in_match
                    total_reward += 1 * num_in_match
                    
                    if chain_level > 1:
                        self.score += 5 * num_in_match
                        total_reward += 5 # Chain reaction bonus

                    # Create particles and mark for removal
                    for item_idx in match_group:
                        item = self.belts[belt_idx][item_idx]
                        items_to_remove_by_id.add(item['id'])
                        y_pos = self.BELT_Y_POS[belt_idx] + self.BELT_HEIGHT / 2
                        self._create_particles(item['x'] + self.ITEM_SIZE / 2, y_pos, self.ITEM_COLORS[item['color_idx']])
            
            if not items_to_remove_by_id:
                break

            # --- Remove matched items and shift remaining items instantly ---
            for i in range(self.NUM_BELTS):
                items_on_belt = self.belts[i]
                
                # Identify gaps created by removed items
                gaps = []
                new_belt = []
                for idx, item in enumerate(items_on_belt):
                    if item['id'] in items_to_remove_by_id:
                        gaps.append(item['x'])
                    else:
                        new_belt.append(item)
                
                if not gaps: continue

                # For each remaining item, calculate how much it needs to shift left
                for item in new_belt:
                    shift_amount = sum(1 for gap_x in gaps if gap_x < item['x']) * (self.ITEM_SIZE + self.ITEM_SPACING)
                    item['x'] -= shift_amount

                self.belts[i] = new_belt

        return total_reward

    def _find_all_matches(self):
        matches = {i: [] for i in range(self.NUM_BELTS)}
        for i in range(self.NUM_BELTS):
            belt = self.belts[i]
            if len(belt) < 3: continue
            
            # Sort belt by x-position to ensure correct adjacency
            sorted_belt = sorted(belt, key=lambda it: it['x'])
            self.belts[i] = sorted_belt

            j = 0
            while j < len(self.belts[i]) - 2:
                color_to_match = self.belts[i][j]['color_idx']
                match_group = [j]
                k = j + 1
                while k < len(self.belts[i]) and self.belts[i][k]['color_idx'] == color_to_match:
                    match_group.append(k)
                    k += 1
                
                if len(match_group) >= 3:
                    matches[i].append(match_group)
                    j = k
                else:
                    j += 1
        return matches

    def _create_item(self, index):
        self.item_id_counter += 1
        return {
            'id': self.item_id_counter,
            'color_idx': self.np_random.integers(0, len(self.ITEM_COLORS)),
            'x': 0, # Will be set later
        }

    def _find_and_fix_initial_matches(self):
        found_match = False
        all_matches = self._find_all_matches()
        for belt_idx, belt_matches in all_matches.items():
            for match_group in belt_matches:
                if match_group:
                    found_match = True
                    # Change the color of the middle item in the match
                    fix_idx = match_group[len(match_group) // 2]
                    original_color = self.belts[belt_idx][fix_idx]['color_idx']
                    new_color = (original_color + 1) % len(self.ITEM_COLORS)
                    self.belts[belt_idx][fix_idx]['color_idx'] = new_color
        return found_match

    def _create_particles(self, x, y, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(20, 80)
            velocity = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(x, y),
                'vel': velocity,
                'lifetime': self.np_random.uniform(0.5, 1.2),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self, dt):
        gravity = pygame.Vector2(0, 150)
        for p in self.particles:
            p['vel'] += gravity * dt
            p['pos'] += p['vel'] * dt
            p['lifetime'] -= dt
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Belts ---
        for y_pos in self.BELT_Y_POS:
            pygame.draw.rect(self.screen, self.COLOR_BELT, (0, y_pos, self.SCREEN_WIDTH, self.BELT_HEIGHT))

        # --- Draw Items ---
        for i in range(self.NUM_BELTS):
            y_pos = self.BELT_Y_POS[i] + (self.BELT_HEIGHT - self.ITEM_SIZE) / 2
            for item in self.belts[i]:
                color = self.ITEM_COLORS[item['color_idx']]
                rect = pygame.Rect(int(item['x']), int(y_pos), self.ITEM_SIZE, self.ITEM_SIZE)
                pygame.draw.rect(self.screen, color, rect, border_radius=4)
                pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), rect, width=2, border_radius=4)

        # --- Draw Cursor and Selection ---
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        
        # Draw selection first
        if self.selected_item_coords:
            sel_belt_idx, sel_item_idx = self.selected_item_coords
            if sel_item_idx < len(self.belts[sel_belt_idx]):
                item = self.belts[sel_belt_idx][sel_item_idx]
                y_pos = self.BELT_Y_POS[sel_belt_idx] + (self.BELT_HEIGHT - self.ITEM_SIZE) / 2
                rect = pygame.Rect(int(item['x']), int(y_pos), self.ITEM_SIZE, self.ITEM_SIZE)
                
                # Pulsing glow effect
                glow_size = int(4 + pulse * 4)
                glow_alpha = int(100 + pulse * 100)
                glow_surf = pygame.Surface((self.ITEM_SIZE + glow_size*2, self.ITEM_SIZE + glow_size*2), pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, self.COLOR_SELECTED + (glow_alpha,), glow_surf.get_rect(), border_radius=8)
                self.screen.blit(glow_surf, (rect.x - glow_size, rect.y - glow_size), special_flags=pygame.BLEND_RGBA_ADD)
                pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect, width=3, border_radius=4)

        # Draw cursor on top
        cur_belt_idx, cur_item_idx = self.cursor_pos
        if cur_item_idx < len(self.belts[cur_belt_idx]):
            item = self.belts[cur_belt_idx][cur_item_idx]
            y_pos = self.BELT_Y_POS[cur_belt_idx] + (self.BELT_HEIGHT - self.ITEM_SIZE) / 2
            rect = pygame.Rect(int(item['x']), int(y_pos), self.ITEM_SIZE, self.ITEM_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, width=3, border_radius=4)

        # --- Draw Particles ---
        for p in self.particles:
            size = p['size'] * (p['lifetime'] / 1.2) # Fade out size
            if size > 1:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(size), p['color'])

    def _render_ui(self):
        # --- Score Display ---
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # --- Timer Display ---
        time_str = f"{int(self.time_remaining // 60):02}:{int(self.time_remaining % 60):02}"
        time_color = (255, 100, 100) if self.time_remaining < 10 else self.COLOR_UI_TEXT
        timer_text = self.font_timer.render(time_str, True, time_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 5))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "cursor_pos": self.cursor_pos,
            "is_holding_item": self.selected_item_coords is not None
        }

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # This block will not be executed in the test environment, but is useful for manual testing.
    # To run, you'll need to `pip install pygame`.
    # It also requires a display, so it won't run in a headless environment.
    # To run it, you might need to comment out the `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` line.
    
    # Re-enable display for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    pygame.display.set_caption("Conveyor Belt Chaos")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    # Map keyboard keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    running = True
    while running:
        # --- Pygame event handling ---
        movement_action = 0
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
        
        keys = pygame.key.get_pressed()
        for key, move in key_to_action.items():
            if keys[key]:
                movement_action = move
                break # Prioritize one move per frame
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset() 
        
        # --- Render to screen ---
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # Transpose back for rendering
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
    env.close()