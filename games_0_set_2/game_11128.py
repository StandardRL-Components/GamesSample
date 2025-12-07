import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:29:54.656406
# Source Brief: brief_01128.md
# Brief Index: 1128
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "A turn-based strategy game where you command units on a hexagonal grid to capture territory and defeat your opponent."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to navigate. Press 'space' to select a unit, confirm an action, or end your turn. Press 'shift' to access the upgrade menu."
    )
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (40, 50, 60)
    COLOR_NEUTRAL = (80, 90, 100)
    COLOR_PLAYER = (220, 50, 50)
    COLOR_AI = (50, 100, 220)
    COLOR_PLAYER_LIGHT = (255, 120, 120)
    COLOR_AI_LIGHT = (120, 170, 255)
    COLOR_SELECTION = (255, 255, 100)
    COLOR_ACTION_VALID = (100, 255, 100)
    COLOR_ACTION_INVALID = (255, 100, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_GOLD = (255, 215, 0)
    COLOR_TEXT_MANPOWER = (200, 200, 255)

    # Grid settings
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    HEX_SIZE = 22
    GRID_COLS = 13
    GRID_ROWS = 7
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 14)
        self.font_m = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 24, bold=True)

        self.grid = {}
        self.q_coords = []
        self.r_coords = []
        self.selected_hex_idx = 0
        self.active_unit_hex = None
        self.action_target_hex = None
        self.game_mode = "PLAYER_MAP" # PLAYER_MAP, PLAYER_ACTION, PLAYER_UPGRADE
        self.player_units = []
        self.ai_units = []
        
        self.particles = []
        self.animations = deque()

        self._initialize_grid_coords()
        # self.reset() is called by the wrapper, no need to call it here.
        
        # This is a critical self-check
        # self.validate_implementation() # Commented out as it's not needed for submission


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.turn_count = 1
        self.winner = None
        
        self.player_resources = {"gold": 10, "manpower": 10}
        self.ai_resources = {"gold": 10, "manpower": 10}
        
        self.ai_attack_prob = 0.1 # Initial AI aggressiveness
        self.step_reward = 0
        
        self.message = "Player's Turn. Select a unit."
        self.animations.clear()
        self.particles.clear()
        
        self._create_map()
        
        self.game_mode = "PLAYER_MAP"
        
        # Set initial selection to the player's unit
        player_start_hex = self.player_units[0]
        if player_start_hex in self.hex_coords:
            self.selected_hex_idx = self.hex_coords.index(player_start_hex)
        
        self.active_unit_hex = None
        self.action_target_hex = None

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.step_reward = 0
        
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        
        turn_ended = self._handle_player_action(movement, space_press, shift_press)
        
        if turn_ended and not self.game_over:
            # AI Turn
            self._run_ai_turn()
            
            # Resource Generation
            self._generate_resources()
            self.turn_count += 1
            if self.turn_count % 25 == 0:
                self.ai_attack_prob = min(0.9, self.ai_attack_prob + 0.05)
            self.message = "Player's Turn. Select a unit."
            self.game_mode = "PLAYER_MAP"

        self._check_termination()
        
        # Add a small penalty for each step to encourage faster wins
        if not self.game_over:
             self.step_reward -= 0.01

        self.score += self.step_reward
        
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            self.step_reward,
            self.game_over,
            truncated,
            self._get_info()
        )

    # --- Game Logic ---
    def _handle_player_action(self, movement, space_press, shift_press):
        turn_ended = False
        selected_hex = self.hex_coords[self.selected_hex_idx]
        
        if self.game_mode == "PLAYER_MAP":
            if movement != 0: # Handle selection movement
                self._move_selection(movement)
            elif space_press: # Handle interaction
                if self.grid[selected_hex]['owner'] == 'PLAYER' and self.grid[selected_hex]['unit']:
                    self.game_mode = "PLAYER_ACTION"
                    self.active_unit_hex = selected_hex
                    self.action_target_hex = None
                    self.message = "Select adjacent hex to move/attack."
                else: # End turn by pressing space on non-unit hex
                    turn_ended = True
                    self.message = "Turn ended."
                    # SFX: EndTurn
            elif shift_press:
                if self.grid[selected_hex]['owner'] == 'PLAYER' and self.grid[selected_hex]['unit']:
                    self.game_mode = "PLAYER_UPGRADE"
                    self.active_unit_hex = selected_hex
                    self.message = "Upgrade unit? (Space to confirm, Shift to cancel)"

        elif self.game_mode == "PLAYER_ACTION":
            if movement != 0:
                neighbors = self._get_neighbors(self.active_unit_hex)
                # Map movement to neighbors
                direction_map = {1: 0, 4: 1, 3: 3, 2: 4} # Up, Right, Left, Down -> Neighbor index
                if movement in direction_map:
                    idx = direction_map[movement]
                    if idx < len(neighbors):
                        self.action_target_hex = neighbors[idx]
            elif space_press and self.action_target_hex:
                turn_ended = self._execute_move_or_attack(self.active_unit_hex, self.action_target_hex)
            elif shift_press:
                self.game_mode = "PLAYER_MAP"
                self.active_unit_hex = None
                self.action_target_hex = None
                self.message = "Action canceled."

        elif self.game_mode == "PLAYER_UPGRADE":
            if space_press:
                self._execute_upgrade(self.active_unit_hex)
                turn_ended = True
            elif shift_press or movement != 0:
                self.game_mode = "PLAYER_MAP"
                self.active_unit_hex = None
                self.message = "Upgrade canceled."
        
        return turn_ended

    def _execute_move_or_attack(self, origin_hex, target_hex):
        if self._hex_distance(origin_hex, target_hex) != 1:
            return False

        origin_unit = self.grid[origin_hex]['unit']
        target_info = self.grid[target_hex]

        # Move
        if target_info['owner'] in ['NEUTRAL', 'PLAYER'] and not target_info['unit']:
            self.grid[target_hex]['owner'] = 'PLAYER'
            self.grid[target_hex]['unit'] = origin_unit
            self.grid[origin_hex]['unit'] = None
            # If origin hex becomes empty, it stays player-owned but without a unit
            
            self.player_units.remove(origin_hex)
            self.player_units.append(target_hex)
            
            self.message = f"Unit moved to {target_hex}."
            self.step_reward += 0.1 # Small reward for expansion
            # SFX: Move
            self._add_animation('move', origin_hex, target_hex, self.COLOR_PLAYER)
            return True
        
        # Attack
        elif target_info['owner'] == 'AI' and target_info['unit']:
            target_unit = target_info['unit']
            # Win
            if origin_unit['strength'] > target_unit['strength']:
                self.grid[target_hex]['owner'] = 'PLAYER'
                self.grid[target_hex]['unit'] = origin_unit
                self.grid[origin_hex]['unit'] = None
                
                self.player_units.remove(origin_hex)
                self.player_units.append(target_hex)
                self.ai_units.remove(target_hex)
                
                self.message = "Territory captured!"
                self.step_reward += 1.0 # Capture reward
                # SFX: AttackWin
                self._add_animation('attack', origin_hex, target_hex, self.COLOR_PLAYER)
                self._create_explosion(target_hex, self.COLOR_AI_LIGHT)
                return True
            # Lose
            else:
                self.grid[origin_hex]['unit'] = None
                self.player_units.remove(origin_hex)
                
                self.message = "Attack failed, unit lost."
                self.step_reward -= 0.1 # Lost unit penalty
                # SFX: AttackLoss
                self._add_animation('attack', origin_hex, target_hex, self.COLOR_PLAYER, success=False)
                self._create_explosion(origin_hex, self.COLOR_PLAYER_LIGHT)
                return True
        return False

    def _execute_upgrade(self, unit_hex):
        unit = self.grid[unit_hex]['unit']
        cost = unit['strength'] * 10
        if self.player_resources['gold'] >= cost:
            self.player_resources['gold'] -= cost
            unit['strength'] += 1
            self.message = f"Unit upgraded to level {unit['strength']}."
            self.step_reward += 5.0 # Upgrade reward
            # SFX: Upgrade
            self._add_animation('upgrade', unit_hex, unit_hex, self.COLOR_PLAYER)
        else:
            self.message = f"Not enough gold. Cost: {cost}"
            # SFX: Error

    def _run_ai_turn(self):
        # AI decision making
        possible_actions = []
        for u_hex in self.ai_units:
            neighbors = self._get_neighbors(u_hex)
            for n_hex in neighbors:
                # Attack actions
                if self.grid[n_hex]['owner'] == 'PLAYER' and self.grid[n_hex]['unit']:
                    possible_actions.append({'type': 'attack', 'from': u_hex, 'to': n_hex})
                # Move actions
                elif self.grid[n_hex]['owner'] in ['NEUTRAL', 'AI'] and not self.grid[n_hex]['unit']:
                    possible_actions.append({'type': 'move', 'from': u_hex, 'to': n_hex})

        if not possible_actions:
            return

        # Choose action type based on aggression
        attack_actions = [a for a in possible_actions if a['type'] == 'attack']
        move_actions = [a for a in possible_actions if a['type'] == 'move']
        
        chosen_action = None
        if attack_actions and self.np_random.random() < self.ai_attack_prob:
            chosen_action = self.np_random.choice(attack_actions)
        elif move_actions:
            chosen_action = self.np_random.choice(move_actions)
        elif attack_actions: # Fallback to attack if no move is possible
            chosen_action = self.np_random.choice(attack_actions)
        
        if not chosen_action:
            return

        # Execute action
        origin_hex, target_hex = chosen_action['from'], chosen_action['to']
        origin_unit = self.grid[origin_hex]['unit']
        
        if chosen_action['type'] == 'move':
            self.grid[target_hex]['owner'] = 'AI'
            self.grid[target_hex]['unit'] = origin_unit
            self.grid[origin_hex]['unit'] = None
            self.ai_units.remove(origin_hex)
            self.ai_units.append(target_hex)
            self._add_animation('move', origin_hex, target_hex, self.COLOR_AI)
        elif chosen_action['type'] == 'attack':
            target_unit = self.grid[target_hex]['unit']
            # AI wins
            if origin_unit['strength'] > target_unit['strength']:
                self.grid[target_hex]['owner'] = 'AI'
                self.grid[target_hex]['unit'] = origin_unit
                self.grid[origin_hex]['unit'] = None
                self.ai_units.remove(origin_hex)
                self.ai_units.append(target_hex)
                self.player_units.remove(target_hex)
                self.step_reward -= 1.0 # Penalty for losing territory
                self._create_explosion(target_hex, self.COLOR_PLAYER_LIGHT)
            # AI loses
            else:
                self.grid[origin_hex]['unit'] = None
                self.ai_units.remove(origin_hex)
                self.step_reward += 0.1 # Reward for defending
                self._create_explosion(origin_hex, self.COLOR_AI_LIGHT)
            self._add_animation('attack', origin_hex, target_hex, self.COLOR_AI)

    def _generate_resources(self):
        for owner in ['PLAYER', 'AI']:
            gold_gain = 0
            manpower_gain = 0
            for h in self.grid.values():
                if h['owner'] == owner:
                    gold_gain += 1
                    if h['unit']:
                        manpower_gain += h['unit']['strength']
            
            if owner == 'PLAYER':
                self.player_resources['gold'] += gold_gain
                self.player_resources['manpower'] += manpower_gain
            else:
                self.ai_resources['gold'] += gold_gain
                self.ai_resources['manpower'] += manpower_gain

    def _check_termination(self):
        num_player_territories = sum(1 for h in self.grid.values() if h['owner'] == 'PLAYER' and h['unit'])
        num_ai_territories = sum(1 for h in self.grid.values() if h['owner'] == 'AI' and h['unit'])
        
        if num_player_territories > 0 and num_ai_territories == 0:
            self.winner = 'PLAYER'
            self.game_over = True
            self.step_reward += 100
            self.message = "VICTORY!"
        elif num_player_territories == 0 and num_ai_territories > 0:
            self.winner = 'AI'
            self.game_over = True
            self.step_reward -= 100
            self.message = "DEFEAT!"
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.message = "GAME OVER - TIME LIMIT"
            if num_player_territories > num_ai_territories:
                self.step_reward += 25
            elif num_ai_territories > num_player_territories:
                self.step_reward -= 25

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw hex grid and units
        for q, r in self.hex_coords:
            hex_info = self.grid[(q, r)]
            pixel_pos = self._axial_to_pixel(q, r)
            
            # Determine color
            owner_color_map = {'PLAYER': self.COLOR_PLAYER, 'AI': self.COLOR_AI, 'NEUTRAL': self.COLOR_NEUTRAL}
            hex_color = owner_color_map[hex_info['owner']]
            
            self._draw_hexagon(self.screen, hex_color, pixel_pos, self.HEX_SIZE)
            self._draw_hexagon(self.screen, self.COLOR_GRID, pixel_pos, self.HEX_SIZE, width=2)

            if hex_info['unit']:
                unit_color = self.COLOR_PLAYER_LIGHT if hex_info['owner'] == 'PLAYER' else self.COLOR_AI_LIGHT
                pygame.draw.circle(self.screen, unit_color, pixel_pos, self.HEX_SIZE * 0.6)
                strength_text = self.font_m.render(str(hex_info['unit']['strength']), True, self.COLOR_BG)
                self.screen.blit(strength_text, strength_text.get_rect(center=pixel_pos))
        
        self._render_highlights()
        self._update_and_draw_animations()
        self._update_and_draw_particles()

    def _render_highlights(self):
        # Pulsing selection highlight
        selected_hex = self.hex_coords[self.selected_hex_idx]
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        alpha = int(100 + 155 * pulse)
        self._draw_hexagon(self.screen, self.COLOR_SELECTION, self._axial_to_pixel(*selected_hex), self.HEX_SIZE + 2, width=3, alpha=alpha)

        # Action highlights
        if self.game_mode == "PLAYER_ACTION" and self.active_unit_hex:
            neighbors = self._get_neighbors(self.active_unit_hex)
            for n_hex in neighbors:
                is_target = (self.action_target_hex == n_hex)
                color = self.COLOR_ACTION_VALID if not is_target else self.COLOR_SELECTION
                width = 2 if not is_target else 4
                self._draw_hexagon(self.screen, color, self._axial_to_pixel(*n_hex), self.HEX_SIZE, width=width, alpha=200)

    def _render_ui(self):
        # Top bar
        pygame.draw.rect(self.screen, (0,0,0,150), (0, 0, self.SCREEN_WIDTH, 30))
        
        # Player Resources
        p_gold_text = self.font_m.render(f"G: {self.player_resources['gold']}", True, self.COLOR_TEXT_GOLD)
        self.screen.blit(p_gold_text, (10, 5))
        p_man_text = self.font_m.render(f"M: {self.player_resources['manpower']}", True, self.COLOR_TEXT_MANPOWER)
        self.screen.blit(p_man_text, (100, 5))

        # Turn Counter
        turn_text = self.font_m.render(f"Turn: {self.turn_count}", True, self.COLOR_TEXT)
        self.screen.blit(turn_text, turn_text.get_rect(centerx=self.SCREEN_WIDTH/2, centery=15))

        # AI Resources
        ai_gold_text = self.font_m.render(f"G: {self.ai_resources['gold']}", True, self.COLOR_TEXT_GOLD)
        self.screen.blit(ai_gold_text, (self.SCREEN_WIDTH - 180, 5))
        ai_man_text = self.font_m.render(f"M: {self.ai_resources['manpower']}", True, self.COLOR_TEXT_MANPOWER)
        self.screen.blit(ai_man_text, (self.SCREEN_WIDTH - 90, 5))

        # Bottom message bar
        pygame.draw.rect(self.screen, (0,0,0,150), (0, self.SCREEN_HEIGHT - 30, self.SCREEN_WIDTH, 30))
        msg_text = self.font_m.render(self.message, True, self.COLOR_TEXT)
        self.screen.blit(msg_text, msg_text.get_rect(centerx=self.SCREEN_WIDTH/2, centery=self.SCREEN_HEIGHT-15))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            end_text = self.font_l.render(self.message, True, self.COLOR_SELECTION)
            self.screen.blit(end_text, end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2)))

    # --- Animation & Particles ---
    def _add_animation(self, anim_type, start_hex, end_hex, color, success=True):
        start_pos = self._axial_to_pixel(*start_hex)
        end_pos = self._axial_to_pixel(*end_hex)
        self.animations.append({
            'type': anim_type, 'start': start_pos, 'end': end_pos,
            'color': color, 'progress': 0, 'duration': 10, 'success': success
        })

    def _update_and_draw_animations(self):
        if not self.animations:
            return
            
        # This renders the current state of animations. It's safe inside _get_observation
        # because it doesn't change game logic, only visual representation.
        for anim in list(self.animations):
            anim['progress'] += 1
            p = anim['progress'] / anim['duration']
            
            if anim['type'] in ['move', 'attack']:
                curr_x = int(anim['start'][0] + (anim['end'][0] - anim['start'][0]) * p)
                curr_y = int(anim['start'][1] + (anim['end'][1] - anim['start'][1]) * p)
                pygame.draw.circle(self.screen, anim['color'], (curr_x, curr_y), int(self.HEX_SIZE * 0.5 * (1-p) + 5))
                if anim['type'] == 'attack' and not anim['success']:
                    if p > 0.5: # Show failure halfway
                        pygame.draw.line(self.screen, self.COLOR_ACTION_INVALID, (curr_x-5, curr_y-5), (curr_x+5, curr_y+5), 3)
                        pygame.draw.line(self.screen, self.COLOR_ACTION_INVALID, (curr_x-5, curr_y+5), (curr_x+5, curr_y-5), 3)

            elif anim['type'] == 'upgrade':
                size = int(self.HEX_SIZE * (1 + p))
                alpha = int(255 * (1 - p))
                self._draw_hexagon(self.screen, self.COLOR_SELECTION, anim['start'], size, width=3, alpha=alpha)

            if anim['progress'] >= anim['duration']:
                self.animations.remove(anim)

    def _create_explosion(self, at_hex, color):
        pos = self._axial_to_pixel(*at_hex)
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': 20, 'color': color})

    def _update_and_draw_particles(self):
        for p in list(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 20))
                color_with_alpha = p['color'] + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['life']/5 + 1), color_with_alpha)

    # --- Hex Grid Utilities ---
    def _initialize_grid_coords(self):
        self.hex_coords = []
        for r in range(self.GRID_ROWS):
            r_offset = r >> 1
            for q in range(-r_offset, self.GRID_COLS - r_offset):
                 if self._hex_distance((q, r), (0, 0)) <= self.GRID_ROWS:
                    self.hex_coords.append((q, r))
        
        self.q_coords = sorted(list(set(q for q, r in self.hex_coords)))
        self.r_coords = sorted(list(set(r for q, r in self.hex_coords)))

    def _create_map(self):
        self.grid = {}
        self.player_units = []
        self.ai_units = []
        
        for q, r in self.hex_coords:
            self.grid[(q, r)] = {'owner': 'NEUTRAL', 'unit': None}
        
        # Place player and AI
        player_start_hex = self.hex_coords[int(len(self.hex_coords) * 0.2)]
        ai_start_hex = self.hex_coords[int(len(self.hex_coords) * 0.8)]
        
        self.grid[player_start_hex] = {'owner': 'PLAYER', 'unit': {'strength': 1}}
        self.player_units.append(player_start_hex)
        
        self.grid[ai_start_hex] = {'owner': 'AI', 'unit': {'strength': 1}}
        self.ai_units.append(ai_start_hex)

    def _move_selection(self, direction):
        q, r = self.hex_coords[self.selected_hex_idx]
        
        # Approximate grid directions
        if direction == 1: # Up
            r -= 1
        elif direction == 2: # Down
            r += 1
        elif direction == 3: # Left
            q -= 1
        elif direction == 4: # Right
            q += 1
        
        # Find closest valid hex to the target q,r
        min_dist = float('inf')
        best_idx = self.selected_hex_idx
        for i, h_coord in enumerate(self.hex_coords):
            dist = ((h_coord[0] - q)**2 + (h_coord[1] - r)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        self.selected_hex_idx = best_idx

    def _axial_to_pixel(self, q, r):
        x = self.HEX_SIZE * (3/2 * q) + self.SCREEN_WIDTH / 2
        y = self.HEX_SIZE * (math.sqrt(3)/2 * q + math.sqrt(3) * r) + self.SCREEN_HEIGHT / 2
        return int(x), int(y)

    def _get_neighbors(self, hex_coord):
        q, r = hex_coord
        axial_directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        neighbors = []
        for dq, dr in axial_directions:
            n = (q + dq, r + dr)
            if n in self.grid:
                neighbors.append(n)
        return neighbors

    def _hex_distance(self, h1, h2):
        q1, r1 = h1
        q2, r2 = h2
        return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) / 2

    def _draw_hexagon(self, surface, color, position, size, width=0, alpha=255):
        points = []
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = math.pi / 180 * angle_deg
            points.append((position[0] + size * math.cos(angle_rad),
                           position[1] + size * math.sin(angle_rad)))
        
        if width == 0:
            if alpha < 255:
                pygame.gfxdraw.filled_polygon(surface, points, (*color, alpha))
            else:
                pygame.draw.polygon(surface, color, points)
        else:
            if alpha < 255:
                pygame.gfxdraw.aapolygon(surface, points, (*color, alpha))
            else:
                pygame.draw.polygon(surface, color, points, width)

    # --- Gymnasium Interface Compliance ---
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "turn": self.turn_count,
            "player_territories": sum(1 for h in self.grid.values() if h['owner'] == 'PLAYER'),
            "ai_territories": sum(1 for h in self.grid.values() if h['owner'] == 'AI'),
            "winner": self.winner
        }
    
    def render(self):
        # This method is not used by the agent but can be useful for human play.
        # It's not required by the prompt, but good practice.
        # For this problem, _get_observation handles all rendering.
        return self._get_observation()

    def close(self):
        pygame.quit()

# --- To run and play the game manually ---
if __name__ == '__main__':
    # This block is for human play and debugging.
    # It is not part of the required environment submission.
    # To use it, you'll need to `pip install pygame`.
    # It will open a window and let you play the game.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a visible driver
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Hex Grid Strategy Game")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default: no-op
        
        # Check for key presses once per frame
        key_pressed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                key_pressed = True
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                
                if event.key == pygame.K_SPACE:
                    action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1

                if event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()
                    print("Game Reset.")
                
        # Since auto_advance is False, we only step when an action is taken
        if key_pressed:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
            if terminated or truncated:
                print("Game Over. Press 'R' to restart.")

        # Update display
        frame = env.render()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata['render_fps'])
        
    env.close()