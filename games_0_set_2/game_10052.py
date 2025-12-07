import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:43:15.206538
# Source Brief: brief_00052.md
# Brief Index: 52
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Gymnasium environment for "Elemental Crystals", a turn-based strategy game.
    The player commands four elemental spirits to capture five crystals on a 5x5 board
    before a 50-turn limit is reached.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Command four elemental spirits to capture five crystals on a 5x5 isometric board. "
        "Use movement, spells, and strategy to win before the 50-turn limit is reached."
    )
    user_guide = (
        "Controls: Use ↑↓←→ arrow keys for movement and menu navigation. "
        "Press 'space' to confirm selections and 'shift' to cancel or go back."
    )
    auto_advance = False

    # --- CONSTANTS ---
    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID_LIGHT = (60, 70, 90)
    COLOR_GRID_DARK = (40, 50, 70)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_DIM = (150, 160, 180)
    COLOR_HIGHLIGHT = (255, 255, 100)
    COLOR_CRYSTAL = (200, 80, 255)
    COLOR_ENEMY = (255, 80, 80)
    
    SPIRIT_COLORS = {
        "FIRE": (255, 100, 50),
        "WATER": (50, 150, 255),
        "EARTH": (100, 200, 80),
        "AIR": (240, 240, 120)
    }

    # Game settings
    BOARD_SIZE = (5, 5)
    MAX_TURNS = 50
    NUM_CRYSTALS = 5
    NUM_PLAYER_SPIRITS = 4
    NUM_ENEMY_SPIRITS = 3
    INGREDIENT_REPLENISH_TURNS = 3

    # Screen dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize state variables
        # self.reset() is called by the environment wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.turn = 1
        self.last_reward = 0

        # Phase management
        self.phase = "SELECT_SPIRIT"
        self.selected_spirit_idx = 0
        self.active_spirit_idx = None
        self.action_menu_idx = 0
        self.cast_selection = []
        self.ingredient_select_idx = 0

        # Action handling
        self.last_action = np.array([0, 0, 0])

        # Place entities
        self._place_entities()
        
        # Ingredients
        self.ingredients = {k: 1 for k in self.SPIRIT_COLORS.keys()}
        self.ingredient_cooldown = self.INGREDIENT_REPLENISH_TURNS

        # Effects
        self.particles = []
        self.bob_offset = 0

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _place_entities(self):
        """Randomly places crystals, player spirits, and enemy spirits on the board."""
        w, h = self.BOARD_SIZE
        all_pos = [(x, y) for x in range(w) for y in range(h)]
        self.np_random.shuffle(all_pos)
        
        # Crystals
        self.crystals = [{'pos': pygame.Vector2(pos), 'owner': None} for pos in all_pos[:self.NUM_CRYSTALS]]
        
        # Player Spirits
        player_starts = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
        spirit_types = list(self.SPIRIT_COLORS.keys())
        self.player_spirits = []
        for i in range(self.NUM_PLAYER_SPIRITS):
            self.player_spirits.append({
                'pos': pygame.Vector2(player_starts[i]),
                'type': spirit_types[i],
                'id': f'P{i}'
            })

        # Enemy Spirits
        occupied_pos = {tuple(c['pos']) for c in self.crystals} | {tuple(s['pos']) for s in self.player_spirits}
        available_pos = [pos for pos in all_pos if pos not in occupied_pos]
        self.np_random.shuffle(available_pos)
        self.enemy_spirits = []
        for i in range(self.NUM_ENEMY_SPIRITS):
            self.enemy_spirits.append({
                'pos': pygame.Vector2(available_pos[i]),
                'id': f'E{i}'
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # Detect button presses (transition from 0 to 1)
        movement = action[0]
        space_pressed = action[1] == 1 and self.last_action[1] == 0
        shift_pressed = action[2] == 1 and self.last_action[2] == 0
        self.last_action = action

        # --- Phase-based Input Handling ---
        if self.phase == "SELECT_SPIRIT":
            reward += self._handle_phase_select_spirit(movement, space_pressed)
        elif self.phase == "ACTION_MENU":
            reward += self._handle_phase_action_menu(movement, space_pressed, shift_pressed)
        elif self.phase == "CAST_SELECT_INGREDIENT_1":
            reward += self._handle_phase_cast_select(movement, space_pressed, shift_pressed, 1)
        elif self.phase == "CAST_SELECT_INGREDIENT_2":
            reward += self._handle_phase_cast_select(movement, space_pressed, shift_pressed, 2)
        elif self.phase == "MOVE":
            reward += self._handle_phase_move(movement)

        self.score += reward
        self.last_reward = reward
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    # --- Phase Handlers ---
    def _handle_phase_select_spirit(self, movement, space_pressed):
        if movement in [1, 2]: # Up/Down
            self.selected_spirit_idx = (self.selected_spirit_idx + (1 if movement == 2 else -1)) % self.NUM_PLAYER_SPIRITS
        if space_pressed:
            self.active_spirit_idx = self.selected_spirit_idx
            self.phase = "ACTION_MENU"
            self.action_menu_idx = 0
            # sfx: UI_Confirm
        return 0

    def _handle_phase_action_menu(self, movement, space_pressed, shift_pressed):
        action_options = ["Move", "Cast Spell", "End Turn"]
        if movement in [1, 2]: # Up/Down
            self.action_menu_idx = (self.action_menu_idx + (1 if movement == 2 else -1)) % len(action_options)
        
        if shift_pressed:
            self.active_spirit_idx = None
            self.phase = "SELECT_SPIRIT"
            # sfx: UI_Cancel
            return 0

        if space_pressed:
            selection = action_options[self.action_menu_idx]
            if selection == "Move":
                self.phase = "MOVE"
                # sfx: UI_Select
            elif selection == "Cast Spell":
                self.cast_selection = []
                self.phase = "CAST_SELECT_INGREDIENT_1"
                self.ingredient_select_idx = 0
                # sfx: UI_Select
            elif selection == "End Turn":
                return self._end_turn() # This returns a reward
        return 0

    def _handle_phase_cast_select(self, movement, space_pressed, shift_pressed, stage):
        available_ingredients = [ing for ing, count in self.ingredients.items() if count > 0]
        if not available_ingredients: # No ingredients to cast
            self.phase = "ACTION_MENU"
            # sfx: UI_Error
            return 0

        if movement in [1, 2]: # Up/Down
            self.ingredient_select_idx = (self.ingredient_select_idx + (1 if movement == 2 else -1)) % len(available_ingredients)

        if shift_pressed:
            self.phase = "ACTION_MENU"
            # sfx: UI_Cancel
            return 0

        if space_pressed:
            selected_ing = available_ingredients[self.ingredient_select_idx]
            self.cast_selection.append(selected_ing)
            self.ingredients[selected_ing] -= 1 # Temporarily consume
            # sfx: UI_Confirm
            
            if stage == 1:
                self.phase = "CAST_SELECT_INGREDIENT_2"
                self.ingredient_select_idx = 0
            else: # Stage 2
                return self._execute_spell() # This ends the turn and returns reward
        return 0

    def _handle_phase_move(self, movement):
        if movement == 0: return 0 # No move
        
        spirit = self.player_spirits[self.active_spirit_idx]
        old_pos = spirit['pos'].copy()
        new_pos = old_pos.copy()
        
        if movement == 1: new_pos.y -= 1 # Up
        elif movement == 2: new_pos.y += 1 # Down
        elif movement == 3: new_pos.x -= 1 # Left
        elif movement == 4: new_pos.x += 1 # Right
        
        # Board boundary check
        if 0 <= new_pos.x < self.BOARD_SIZE[0] and 0 <= new_pos.y < self.BOARD_SIZE[1]:
            # Collision check with other spirits
            all_spirit_pos = {tuple(s['pos']) for s in self.player_spirits} | {tuple(s['pos']) for s in self.enemy_spirits}
            if tuple(new_pos) not in all_spirit_pos:
                
                # --- Movement Reward Calculation ---
                reward = 0
                dist_before = min([old_pos.distance_to(c['pos']) for c in self.crystals if c['owner'] != 'PLAYER'] or [0])
                dist_after = min([new_pos.distance_to(c['pos']) for c in self.crystals if c['owner'] != 'PLAYER'] or [0])
                
                if dist_after < dist_before:
                    reward += 1.0 # Closer to a crystal
                elif dist_after > dist_before:
                    reward -= 0.1 # Further from a crystal

                spirit['pos'] = new_pos
                # sfx: Spirit_Move
                return reward + self._end_turn()
        
        # If move is invalid, stay in MOVE phase
        # sfx: UI_Error
        return 0

    def _end_turn(self):
        """Finalizes a turn, runs enemy AI, checks win/loss conditions."""
        end_of_turn_reward = 0
        
        self._handle_enemy_turn()
        end_of_turn_reward += self._update_crystal_ownership()

        # Replenish ingredients
        self.ingredient_cooldown -= 1
        if self.ingredient_cooldown <= 0:
            self.ingredients = {k: 1 for k in self.SPIRIT_COLORS.keys()}
            self.ingredient_cooldown = self.INGREDIENT_REPLENISH_TURNS
            # sfx: Ingredients_Replenish

        self.turn += 1
        
        # Check termination conditions
        if all(c['owner'] == 'PLAYER' for c in self.crystals):
            self.game_over = True
            end_of_turn_reward += 100 # Victory
        elif self.turn > self.MAX_TURNS:
            self.game_over = True
            end_of_turn_reward -= 50 # Timeout

        # Reset for next player turn
        self.phase = "SELECT_SPIRIT"
        self.active_spirit_idx = None
        self.selected_spirit_idx = 0
        
        return end_of_turn_reward

    def _execute_spell(self):
        """Executes a spell based on selected ingredients and ends the turn."""
        spell_reward = 0
        # sfx: Spell_Cast_Generic
        
        # Sort to make combinations order-independent
        combo = tuple(sorted(self.cast_selection))
        spirit = self.player_spirits[self.active_spirit_idx]
        
        # --- Spell Definitions ---
        if combo == ('FIRE', 'WATER'): # Steam Blast: Push adjacent enemies
            for enemy in self.enemy_spirits:
                if spirit['pos'].distance_to(enemy['pos']) < 1.5: # 1-tile radius
                    direction = (enemy['pos'] - spirit['pos']).normalize()
                    new_pos = enemy['pos'] + direction
                    if 0 <= new_pos.x < self.BOARD_SIZE[0] and 0 <= new_pos.y < self.BOARD_SIZE[1]:
                        enemy['pos'] = new_pos
                    self._create_particles(spirit['pos'], self.SPIRIT_COLORS['WATER'], 20)

        elif combo == ('EARTH', 'EARTH'): # Rock Wall: Not implemented for simplicity, acts as a fizzle
             self._create_particles(spirit['pos'], self.SPIRIT_COLORS['EARTH'], 10, speed=0.5)

        elif combo == ('AIR', 'FIRE'): # Fireball: Damage (remove) closest enemy
            closest_enemy, min_dist = None, float('inf')
            for enemy in self.enemy_spirits:
                dist = spirit['pos'].distance_to(enemy['pos'])
                if dist < min_dist:
                    min_dist = dist
                    closest_enemy = enemy
            if closest_enemy:
                self._create_particles(closest_enemy['pos'], self.SPIRIT_COLORS['FIRE'], 50)
                self.enemy_spirits.remove(closest_enemy)
                spell_reward += 10 # Reward for defeating an enemy
                # sfx: Enemy_Defeated

        else: # Fizzle - ingredients are consumed but nothing happens
            self._create_particles(spirit['pos'], (100,100,100), 10)
            # sfx: Spell_Fizzle
            
        # Refund unused ingredients if spell fails, otherwise they are consumed
        self.cast_selection = []
        return spell_reward + self._end_turn()
        
    def _handle_enemy_turn(self):
        """Moves each enemy spirit to a random valid adjacent tile."""
        all_spirit_pos = {tuple(s['pos']) for s in self.player_spirits} | {tuple(s['pos']) for s in self.enemy_spirits}
        for enemy in self.enemy_spirits:
            moves = [pygame.Vector2(0, -1), pygame.Vector2(0, 1), pygame.Vector2(-1, 0), pygame.Vector2(1, 0)]
            self.np_random.shuffle(moves)
            for move in moves:
                new_pos = enemy['pos'] + move
                if (0 <= new_pos.x < self.BOARD_SIZE[0] and 
                    0 <= new_pos.y < self.BOARD_SIZE[1] and 
                    tuple(new_pos) not in all_spirit_pos):
                    enemy['pos'] = new_pos
                    all_spirit_pos.add(tuple(new_pos)) # Prevent other enemies from moving to the same new spot
                    break
    
    def _update_crystal_ownership(self):
        """Checks and updates crystal ownership, returning associated rewards."""
        reward = 0
        player_positions = {tuple(s['pos']) for s in self.player_spirits}
        enemy_positions = {tuple(s['pos']) for s in self.enemy_spirits}

        for crystal in self.crystals:
            pos = tuple(crystal['pos'])
            was_player_owned = crystal['owner'] == 'PLAYER'
            is_player_on = pos in player_positions
            is_enemy_on = pos in enemy_positions

            new_owner = None
            if is_player_on:
                new_owner = 'PLAYER'
            elif is_enemy_on:
                new_owner = 'ENEMY'
            
            if new_owner == 'PLAYER' and not was_player_owned:
                reward += 5  # Capture reward
                # sfx: Crystal_Capture
            elif new_owner != 'PLAYER' and was_player_owned:
                reward -= 1  # Lost crystal penalty
                # sfx: Crystal_Lost
            
            crystal['owner'] = new_owner
        return reward

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Update and render all game elements
        self._update_and_render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "turn": self.turn,
            "phase": self.phase,
            "last_reward": self.last_reward,
            "crystals_captured": sum(1 for c in self.crystals if c['owner'] == 'PLAYER')
        }

    # --- Rendering ---
    def _grid_to_screen(self, x, y):
        """Converts 5x5 grid coordinates to screen coordinates for isometric view."""
        tile_w, tile_h = 64, 32
        origin_x = self.SCREEN_WIDTH / 2
        origin_y = 120
        screen_x = origin_x + (x - y) * tile_w / 2
        screen_y = origin_y + (x + y) * tile_h / 2
        return int(screen_x), int(screen_y)

    def _update_and_render_game(self):
        self.bob_offset = math.sin(self.steps * 0.1) * 3

        # Draw grid
        for y in range(self.BOARD_SIZE[1]):
            for x in range(self.BOARD_SIZE[0]):
                color = self.COLOR_GRID_LIGHT if (x + y) % 2 == 0 else self.COLOR_GRID_DARK
                self._draw_iso_tile(self.screen, color, x, y)
        
        # Collect all drawable entities to sort by y-order for correct overlap
        entities = []
        entities.extend([{'type': 'crystal', **c} for c in self.crystals])
        entities.extend([{'type': 'spirit', **s} for s in self.player_spirits])
        entities.extend([{'type': 'spirit', **s} for s in self.enemy_spirits])
        
        entities.sort(key=lambda e: e['pos'].y * 100 + e['pos'].x)

        for entity in entities:
            if entity['type'] == 'crystal':
                self._draw_crystal(entity)
            elif entity['type'] == 'spirit':
                self._draw_spirit(entity)

        # Draw selection cursors on top
        self._draw_cursors()

        # Update and draw particles
        self._update_particles()

    def _draw_iso_tile(self, surface, color, x, y):
        tile_w, tile_h = 64, 32
        px, py = self._grid_to_screen(x, y)
        points = [
            (px, py),
            (px + tile_w / 2, py + tile_h / 2),
            (px, py + tile_h),
            (px - tile_w / 2, py + tile_h / 2)
        ]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.aalines(surface, self.COLOR_GRID_DARK, True, points)

    def _draw_crystal(self, crystal):
        px, py = self._grid_to_screen(crystal['pos'].x, crystal['pos'].y)
        py += int(self.bob_offset) - 10
        
        color = self.COLOR_CRYSTAL
        if crystal['owner'] == 'PLAYER':
            color = self.SPIRIT_COLORS['AIR']
        elif crystal['owner'] == 'ENEMY':
            color = self.COLOR_ENEMY

        # Simple crystal shape
        points = [
            (px, py - 12), (px + 8, py), (px - 8, py)
        ]
        pygame.draw.polygon(self.screen, color, points)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _draw_spirit(self, spirit):
        px, py = self._grid_to_screen(spirit['pos'].x, spirit['pos'].y)
        py += int(self.bob_offset) - 15
        radius = 12

        # Shadow
        shadow_rect = pygame.Rect(px - radius, py + radius - 2, radius * 2, 8)
        shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, (0, 0, 0, 50), (0, 0, shadow_rect.width, shadow_rect.height))
        self.screen.blit(shadow_surface, shadow_rect.topleft)

        # Body
        color = self.COLOR_ENEMY if 'E' in spirit['id'] else self.SPIRIT_COLORS[spirit['type']]
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius, color)
        pygame.gfxdraw.aacircle(self.screen, px, py, radius, color)

        # Highlight
        highlight_color = tuple(min(255, c + 80) for c in color)
        pygame.gfxdraw.filled_circle(self.screen, px - 4, py - 4, 3, highlight_color)

    def _draw_cursors(self):
        # Draw cursor over the currently selected spirit for movement/action
        if self.phase == "SELECT_SPIRIT":
            spirit = self.player_spirits[self.selected_spirit_idx]
            px, py = self._grid_to_screen(spirit['pos'].x, spirit['pos'].y)
            pygame.draw.circle(self.screen, self.COLOR_HIGHLIGHT, (px, py + 10), 20, 2)
        
        # Draw cursor under the active spirit
        if self.active_spirit_idx is not None:
            spirit = self.player_spirits[self.active_spirit_idx]
            px, py = self._grid_to_screen(spirit['pos'].x, spirit['pos'].y)
            pygame.draw.circle(self.screen, (255, 255, 255), (px, py + 15), 18, 1)

    def _create_particles(self, pos, color, count, speed=2):
        px, py = self._grid_to_screen(pos.x, pos.y)
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(0.5, 1) * speed
            self.particles.append({
                'pos': pygame.Vector2(px, py),
                'vel': vel,
                'lifetime': random.randint(20, 40),
                'color': color
            })
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)
            else:
                size = max(1, int(p['lifetime'] / 8))
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), size)

    def _render_ui(self):
        # --- Top Bar ---
        turn_text = f"Turn: {self.turn}/{self.MAX_TURNS}"
        score_text = f"Score: {int(self.score)}"
        crystals_text = f"Crystals: {sum(1 for c in self.crystals if c['owner'] == 'PLAYER')}/{self.NUM_CRYSTALS}"
        self._draw_text(turn_text, (10, 10))
        self._draw_text(score_text, (self.SCREEN_WIDTH - 150, 10))
        self._draw_text(crystals_text, (self.SCREEN_WIDTH / 2 - 80, 10))
        
        # --- Bottom Bar / Phase Info ---
        prompt = ""
        if self.phase == "SELECT_SPIRIT":
            prompt = f"PHASE: SELECT SPIRIT [↑↓ to cycle, SPACE to confirm]"
        elif self.phase == "ACTION_MENU":
            prompt = f"PHASE: ACTION MENU (Spirit: {self.player_spirits[self.active_spirit_idx]['type']}) [↑↓, SPACE, SHIFT to cancel]"
            self._draw_action_menu()
        elif self.phase == "MOVE":
            prompt = "PHASE: MOVE [Arrow keys to move]"
        elif "CAST_SELECT" in self.phase:
            prompt = f"PHASE: SELECT INGREDIENT {self.phase[-1]} [↑↓, SPACE, SHIFT to cancel]"
            self._draw_ingredient_menu()

        self._draw_text(prompt, (10, self.SCREEN_HEIGHT - 30), font=self.font_small, color=self.COLOR_TEXT_DIM)

        # --- Ingredient List ---
        ing_y = 50
        self._draw_text("Ingredients:", (10, ing_y))
        for i, (name, count) in enumerate(self.ingredients.items()):
            color = self.SPIRIT_COLORS[name] if count > 0 else self.COLOR_TEXT_DIM
            self._draw_text(f"{name}: {count}", (20, ing_y + 25 * (i + 1)), color=color)

    def _draw_action_menu(self):
        if self.active_spirit_idx is None: return
        spirit = self.player_spirits[self.active_spirit_idx]
        px, py = self._grid_to_screen(spirit['pos'].x, spirit['pos'].y)
        
        options = ["Move", "Cast Spell", "End Turn"]
        for i, opt in enumerate(options):
            color = self.COLOR_HIGHLIGHT if i == self.action_menu_idx else self.COLOR_TEXT
            self._draw_text(opt, (px + 40, py - 30 + i * 20), color=color)

    def _draw_ingredient_menu(self):
        if self.active_spirit_idx is None: return
        spirit = self.player_spirits[self.active_spirit_idx]
        px, py = self._grid_to_screen(spirit['pos'].x, spirit['pos'].y)

        available_ingredients = [ing for ing, count in self.ingredients.items() if count > 0]
        if not available_ingredients: return

        for i, ing in enumerate(available_ingredients):
            color = self.COLOR_HIGHLIGHT if i == self.ingredient_select_idx else self.SPIRIT_COLORS[ing]
            self._draw_text(ing, (px + 40, py - 30 + i * 20), color=color)

    def _draw_text(self, text, pos, font=None, color=None):
        if font is None: font = self.font_main
        if color is None: color = self.COLOR_TEXT
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Elemental Crystals")
        clock = pygame.time.Clock()
        
        terminated = False
        running = True
        
        # Store held status for keys
        key_state = {
            'up': 0, 'down': 0, 'left': 0, 'right': 0,
            'space': 0, 'shift': 0
        }
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: key_state['up'] = 1
                    if event.key == pygame.K_DOWN: key_state['down'] = 1
                    if event.key == pygame.K_LEFT: key_state['left'] = 1
                    if event.key == pygame.K_RIGHT: key_state['right'] = 1
                    if event.key == pygame.K_SPACE: key_state['space'] = 1
                    if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: key_state['shift'] = 1
                    if event.key == pygame.K_r: # Reset game
                        obs, info = env.reset()
                        terminated = False
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_UP: key_state['up'] = 0
                    if event.key == pygame.K_DOWN: key_state['down'] = 0
                    if event.key == pygame.K_LEFT: key_state['left'] = 0
                    if event.key == pygame.K_RIGHT: key_state['right'] = 0
                    if event.key == pygame.K_SPACE: key_state['space'] = 0
                    if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: key_state['shift'] = 0

            # Construct action from key state
            movement_action = 0 # None
            if key_state['up']: movement_action = 1
            elif key_state['down']: movement_action = 2
            elif key_state['left']: movement_action = 3
            elif key_state['right']: movement_action = 4
            
            space_action = 1 if key_state['space'] else 0
            shift_action = 1 if key_state['shift'] else 0
            
            action = [movement_action, space_action, shift_action]
            
            # Step the environment
            if not terminated:
                obs, reward, terminated, truncated, info = env.step(action)
                if reward != 0:
                    print(f"Step: {info['steps']}, Turn: {info['turn']}, Phase: {info['phase']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")
                if terminated:
                    print("--- GAME OVER ---")
                    print(f"Final Score: {info['score']}")

            # Render the observation to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Run at 30 FPS

        env.close()
    except pygame.error as e:
        print(f"Could not run interactive test: {e}")
        print("This is expected in a headless environment.")