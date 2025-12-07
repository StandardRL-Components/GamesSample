import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
from collections import Counter
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A turn-based strategy game where players race to collect artifacts by "
        "crafting spells from elemental dice."
    )
    user_guide = (
        "Controls: Use ↑↓ to select a spell and ←→ to select a target. "
        "Press space to cast. Hold shift while casting to clone a known spell for a power boost."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array", num_players=4, num_artifacts=5):
        super().__init__()

        # --- Critical Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Game Configuration ---
        self.render_mode = render_mode
        self.width, self.height = 640, 400
        self.num_players = num_players
        self.num_artifacts = num_artifacts
        self.max_steps = 1000

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 18)
        self.font_m = pygame.font.Font(None, 24)
        self.font_l = pygame.font.Font(None, 48)

        # --- Color Palette ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_DIM = (150, 150, 170)
        self.PLAYER_COLORS = [(255, 80, 80), (80, 120, 255), (80, 255, 120), (255, 200, 80)]
        self.ARTIFACT_COLORS = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (255, 128, 0), (0, 255, 0)]
        self.ELEMENT_COLORS = {
            "FIRE": (255, 100, 50), "ICE": (100, 200, 255), "NATURE": (100, 255, 100),
            "LIGHT": (255, 255, 150), "DARK": (180, 100, 255), "WILD": (255, 255, 255)
        }

        # --- Game Mechanics ---
        self._define_spellbook()
        self.board_nodes = self._create_board_layout(12, 7, 50, 50)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.players = []
        self.artifacts = []
        self.particles = []
        self.current_turn = 0
        self.game_phase = ""
        self.dice_results = []
        self.available_spells = []
        self.selected_spell_idx = 0
        self.selected_target_idx = 0
        self.action_taken_this_turn = False
        self.turn_timer = 0
        self.last_action_reward = 0
        self.winner = -1

    def _define_spellbook(self):
        self.spellbook = [
            {'name': 'Dash', 'recipe': Counter({'LIGHT': 1}), 'target': 'self', 'power': 1, 'desc': 'Move +1 space'},
            {'name': 'Leap', 'recipe': Counter({'NATURE': 1}), 'target': 'self', 'power': 2, 'desc': 'Move +2 spaces'},
            {'name': 'Sprint', 'recipe': Counter({'LIGHT': 2}), 'target': 'self', 'power': 3, 'desc': 'Move +3 spaces'},
            {'name': 'Push', 'recipe': Counter({'FIRE': 1}), 'target': 'other', 'power': -1, 'desc': 'Target moves -1'},
            {'name': 'Chill', 'recipe': Counter({'ICE': 1}), 'target': 'other', 'power': -2, 'desc': 'Target moves -2'},
            {'name': 'Tele-swap', 'recipe': Counter({'DARK': 2}), 'target': 'other', 'power': 0, 'desc': 'Swap with target'},
            {'name': 'Wild Surge', 'recipe': Counter({'WILD': 1}), 'target': 'self', 'power': random.randint(1, 4), 'desc': 'Move +? spaces'},
            {'name': 'Fireball', 'recipe': Counter({'FIRE': 2}), 'target': 'other', 'power': -3, 'desc': 'Target moves -3'},
            {'name': 'Genesis', 'recipe': Counter({'NATURE': 2, 'LIGHT': 1}), 'target': 'self', 'power': 4, 'desc': 'Move +4 spaces'},
        ]

    def _create_board_layout(self, cols, rows, x_offset, y_offset):
        nodes = []
        for r in range(rows):
            for c in range(cols):
                x = x_offset + c * ((self.width - x_offset * 2) / (cols - 1))
                y = y_offset + r * ((self.height - y_offset * 2) / (rows - 1))
                nodes.append(pygame.Vector2(x, y))
        return nodes

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.winner = -1
        self.particles.clear()

        start_nodes = random.sample(range(len(self.board_nodes)), self.num_players)
        self.players = []
        for i in range(self.num_players):
            node_idx = start_nodes[i]
            self.players.append({
                'id': i,
                'pos': pygame.Vector2(self.board_nodes[node_idx]),
                'target_pos': pygame.Vector2(self.board_nodes[node_idx]),
                'node_idx': node_idx,
                'color': self.PLAYER_COLORS[i],
                'artifacts': set(),
                'known_spells': {},
                'move_bonus': 0,
                'is_moving': False
            })

        artifact_nodes = random.sample([i for i in range(len(self.board_nodes)) if i not in start_nodes], self.num_artifacts)
        self.artifacts = []
        for i in range(self.num_artifacts):
            node_idx = artifact_nodes[i]
            self.artifacts.append({
                'id': i,
                'pos': self.board_nodes[node_idx],
                'node_idx': node_idx,
                'color': self.ARTIFACT_COLORS[i],
                'collected_by': -1
            })

        self.current_turn = 0
        self._start_turn()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        self.last_action_reward = 0
        self.game_over = self._check_termination()

        if not self.game_over:
            self._update_game_state(action)
        
        self._update_animations()

        reward = self.last_action_reward
        if self.game_over:
            if self.winner == 0:
                reward += 100
            elif self.winner != -1: # Agent lost
                reward -= 100
        
        self.score += reward
        
        terminated = self.game_over
        truncated = self.steps >= self.max_steps
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_game_state(self, action):
        if self.game_phase == 'AWAITING_INPUT' and self.current_turn == 0 and not self.action_taken_this_turn:
            self._handle_player_input(action)
        elif self.game_phase == 'AI_TURN':
            self.turn_timer -= 1
            if self.turn_timer <= 0:
                self._handle_ai_turn()
        elif self.game_phase == 'MOVEMENT':
            moving_player = self.players[self.current_turn]
            if not moving_player['is_moving']:
                self._end_turn()

    def _start_turn(self):
        self.action_taken_this_turn = False
        self.dice_results = [random.choice(list(self.ELEMENT_COLORS.keys())) for _ in range(3)]
        # sfx: dice roll
        self.available_spells = self._find_available_spells()
        self.selected_spell_idx = 0
        self.selected_target_idx = 0

        if not self.available_spells:
             self._end_turn() # Skip turn if no spells can be crafted
             return

        if self.current_turn == 0:
            self.game_phase = 'AWAITING_INPUT'
        else:
            self.game_phase = 'AI_TURN'
            self.turn_timer = 15 # AI "thinking" time in frames

    def _handle_player_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if self.available_spells:
            if movement == 1: self.selected_spell_idx = (self.selected_spell_idx - 1) % len(self.available_spells)
            if movement == 2: self.selected_spell_idx = (self.selected_spell_idx + 1) % len(self.available_spells)
            
            spell = self.available_spells[self.selected_spell_idx]
            if spell['target'] == 'other':
                num_targets = self.num_players - 1
                if num_targets > 0:
                    if movement == 3: self.selected_target_idx = (self.selected_target_idx - 1) % num_targets
                    if movement == 4: self.selected_target_idx = (self.selected_target_idx + 1) % num_targets

        if space_held and self.available_spells:
            self.action_taken_this_turn = True
            # sfx: spell select
            self._cast_spell(self.current_turn, shift_held)

    def _handle_ai_turn(self):
        # Simple AI: find best move to get to the closest artifact
        ai_player = self.players[self.current_turn]
        
        # Find closest uncollected artifact
        uncollected_artifacts = [a for a in self.artifacts if a['collected_by'] == -1]
        if not uncollected_artifacts:
            self._end_turn()
            return

        closest_artifact = min(uncollected_artifacts, key=lambda a: ai_player['pos'].distance_to(a['pos']))
        
        # Evaluate available spells
        best_spell_idx = -1
        best_target_idx = -1
        best_outcome_dist = float('inf')

        for i, spell in enumerate(self.available_spells):
            if spell['target'] == 'self':
                move_power = spell['power']
                # Simulate move
                new_pos = self._get_closest_node(ai_player['node_idx'], move_power, closest_artifact['node_idx'])
                dist = new_pos.distance_to(closest_artifact['pos'])
                if dist < best_outcome_dist:
                    best_outcome_dist = dist
                    best_spell_idx = i
                    best_target_idx = 0
            # AI doesn't use offensive spells for simplicity
        
        if best_spell_idx != -1:
            self.selected_spell_idx = best_spell_idx
            self.selected_target_idx = best_target_idx
            self._cast_spell(self.current_turn, False) # AI doesn't clone
        else: # No beneficial spell found
            self._end_turn()

    def _cast_spell(self, caster_id, is_clone):
        spell_info = self.available_spells[self.selected_spell_idx]
        caster = self.players[caster_id]

        power = spell_info['power']
        spell_name = spell_info['name']

        if is_clone and spell_name in caster['known_spells']:
            # sfx: clone spell
            power = int(power * 1.5) # Cloned spells are 50% stronger
            self._create_particles(caster['pos'], self.ELEMENT_COLORS['WILD'], 30, 1.5)
        else:
            # sfx: cast spell
            caster['known_spells'][spell_name] = spell_info # Learn spell
            self._create_particles(caster['pos'], self.ELEMENT_COLORS[spell_info['recipe'].most_common(1)[0][0]], 20)

        if spell_info['target'] == 'self':
            self._apply_spell_effect(caster, caster, spell_info, power)
        else: # target == 'other'
            target_options = [p for p in self.players if p['id'] != caster_id]
            if target_options:
                target = target_options[self.selected_target_idx]
                self._apply_spell_effect(caster, target, spell_info, power)
                if caster_id == 0: # Agent affected an opponent
                    self.last_action_reward += 1
                if target['id'] == 0: # Agent was affected
                    self.last_action_reward -= 2

        self.game_phase = 'MOVEMENT'

    def _apply_spell_effect(self, caster, target, spell, power):
        if spell['name'] == 'Tele-swap':
            caster_node, target_node = caster['node_idx'], target['node_idx']
            caster['node_idx'], target['node_idx'] = target_node, caster_node
            caster['target_pos'] = self.board_nodes[caster['node_idx']]
            target['target_pos'] = self.board_nodes[target['node_idx']]
            caster['is_moving'] = True
            target['is_moving'] = True
        else: # Movement-based spells
            move_amount = power + target['move_bonus']
            target['move_bonus'] = 0
            
            old_dist_to_artifact = float('inf')
            if target['id'] == 0:
                uncollected = [a for a in self.artifacts if a['collected_by'] == -1]
                if uncollected:
                    closest = min(uncollected, key=lambda a: target['pos'].distance_to(a['pos']))
                    old_dist_to_artifact = target['pos'].distance_to(closest['pos'])
            
            # Find best node to move to
            target_artifact = None
            uncollected = [a for a in self.artifacts if a['collected_by'] == -1]
            if uncollected:
                target_artifact = min(uncollected, key=lambda a: target['pos'].distance_to(a['pos']))

            target_node_idx = target['node_idx']
            if target_artifact:
                target_node_idx = target_artifact['node_idx']

            new_node_pos = self._get_closest_node(target['node_idx'], move_amount, target_node_idx)
            new_node_idx = self.board_nodes.index(new_node_pos)

            target['node_idx'] = new_node_idx
            target['target_pos'] = new_node_pos
            target['is_moving'] = True

            if target['id'] == 0 and target_artifact:
                new_dist_to_artifact = new_node_pos.distance_to(target_artifact['pos'])
                if new_dist_to_artifact < old_dist_to_artifact:
                    self.last_action_reward += 1.0 # Closer to artifact
                else:
                    self.last_action_reward -= 0.1 # Further from artifact

    def _end_turn(self):
        # Check for artifact collection
        player = self.players[self.current_turn]
        for artifact in self.artifacts:
            if artifact['collected_by'] == -1 and player['node_idx'] == artifact['node_idx']:
                artifact['collected_by'] = player['id']
                player['artifacts'].add(artifact['id'])
                # sfx: artifact collect
                self._create_particles(player['pos'], artifact['color'], 50, 2.0)
                if player['id'] == 0:
                    self.last_action_reward += 5

        if self._check_termination():
            self.game_over = True
            return

        self.current_turn = (self.current_turn + 1) % self.num_players
        self._start_turn()

    def _get_closest_node(self, start_node_idx, num_steps, target_node_idx):
        # Simplified movement: find node that is 'num_steps' away along the line to target
        start_pos = self.board_nodes[start_node_idx]
        target_pos = self.board_nodes[target_node_idx]
        
        if num_steps == 0 or start_pos == target_pos:
            return start_pos

        direction = (target_pos - start_pos)
        if direction.length() > 0:
            direction = direction.normalize()

        # approx dist between nodes
        step_dist = self.board_nodes[0].distance_to(self.board_nodes[1])
        
        final_pos = start_pos + direction * num_steps * step_dist

        # Find the actual board node closest to this theoretical final position
        closest_node = min(self.board_nodes, key=lambda node: node.distance_to(final_pos))
        return closest_node

    def _find_available_spells(self):
        dice_counts = Counter(self.dice_results)
        available = []
        for spell in self.spellbook:
            if not (spell['recipe'] - dice_counts):
                available.append(spell)
        return available

    def _check_termination(self):
        if self.steps >= self.max_steps:
            return True
        for i, player in enumerate(self.players):
            if len(player['artifacts']) >= self.num_artifacts:
                self.winner = i
                return True
        return False

    def _update_animations(self):
        # Player movement interpolation
        for player in self.players:
            if player['is_moving']:
                target_pos = player['target_pos']
                current_pos = player['pos']
                speed = 5.0

                direction_vec = target_pos - current_pos
                distance = direction_vec.length()

                if distance <= speed:
                    player['pos'] = pygame.Vector2(target_pos)
                    player['is_moving'] = False
                else:
                    player['pos'] += direction_vec.normalize() * speed
        
        # Particle updates
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'life': random.randint(20, 40),
                'color': color
            })
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "current_turn": self.current_turn,
            "game_phase": self.game_phase,
            "winner": self.winner,
        }

    def _render_game(self):
        # Draw grid
        for n1 in self.board_nodes:
            for n2 in self.board_nodes:
                if n1 != n2 and n1.distance_to(n2) < 65: # Connect close nodes
                    pygame.draw.aaline(self.screen, self.COLOR_GRID, n1, n2)

        # Draw artifacts
        for artifact in self.artifacts:
            if artifact['collected_by'] == -1:
                pygame.gfxdraw.filled_circle(self.screen, int(artifact['pos'].x), int(artifact['pos'].y), 10, self.COLOR_GRID)
                pygame.gfxdraw.filled_circle(self.screen, int(artifact['pos'].x), int(artifact['pos'].y), 8, artifact['color'])
                pygame.gfxdraw.aacircle(self.screen, int(artifact['pos'].x), int(artifact['pos'].y), 8, (255,255,255))


        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * 6)))
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, 2, 2, 2, (*p['color'], alpha))
            self.screen.blit(temp_surf, (int(p['pos'].x) - 2, int(p['pos'].y) - 2))


        # Draw players
        for player in sorted(self.players, key=lambda p: p['pos'].y): # Draw bottom players first
            p_pos = (int(player['pos'].x), int(player['pos'].y))
            
            # Create a temporary surface for alpha blending
            glow_size = 18 + int(abs(math.sin(self.steps * 0.1)) * 6) if self.current_turn == player['id'] else 18
            glow_alpha = 80 + int(abs(math.sin(self.steps*0.1))*40) if self.current_turn == player['id'] else 50
            glow_color = player['color'] + (glow_alpha,)
            
            temp_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, glow_size, glow_size, glow_size, glow_color)
            self.screen.blit(temp_surf, (p_pos[0] - glow_size, p_pos[1] - glow_size))

            pygame.gfxdraw.filled_circle(self.screen, p_pos[0], p_pos[1], 12, self.COLOR_BG)
            pygame.gfxdraw.filled_circle(self.screen, p_pos[0], p_pos[1], 10, player['color'])
            pygame.gfxdraw.aacircle(self.screen, p_pos[0], p_pos[1], 10, self.COLOR_BG)

            # Draw collected artifacts above player
            for i, art_id in enumerate(player['artifacts']):
                art_color = self.artifacts[art_id]['color']
                pygame.gfxdraw.filled_circle(self.screen, p_pos[0] - 15 + i * 15, p_pos[1] - 20, 5, art_color)
                pygame.gfxdraw.aacircle(self.screen, p_pos[0] - 15 + i * 15, p_pos[1] - 20, 5, self.COLOR_BG)


    def _render_ui(self):
        # Turn indicator
        turn_text = f"Turn: Player {self.current_turn + 1}"
        text_surf = self.font_m.render(turn_text, True, self.PLAYER_COLORS[self.current_turn])
        self.screen.blit(text_surf, (10, 10))

        # Score
        score_text = f"Score: {int(self.score)}"
        text_surf = self.font_m.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.width - text_surf.get_width() - 10, 10))

        # Player 0's turn UI
        if self.current_turn == 0 and self.game_phase == 'AWAITING_INPUT':
            # Dice roll
            dice_title = self.font_m.render("Dice Roll:", True, self.COLOR_TEXT)
            self.screen.blit(dice_title, (10, self.height - 80))
            for i, element in enumerate(self.dice_results):
                color = self.ELEMENT_COLORS[element]
                pygame.draw.rect(self.screen, color, (110 + i * 40, self.height - 80, 30, 30), border_radius=5)
            
            # Spell list
            spell_title = self.font_m.render("Craft Spell (Up/Down):", True, self.COLOR_TEXT)
            self.screen.blit(spell_title, (10, self.height - 45))
            
            if self.available_spells:
                for i, spell in enumerate(self.available_spells):
                    is_selected = (i == self.selected_spell_idx)
                    color = self.COLOR_TEXT if is_selected else self.COLOR_TEXT_DIM
                    text = f"{spell['name']} - {spell['desc']}"
                    
                    if is_selected:
                        # Use a temporary surface for alpha blending
                        rect_surf = pygame.Surface((300, 20), pygame.SRCALPHA)
                        rect_surf.fill(self.PLAYER_COLORS[0] + (50,))
                        self.screen.blit(rect_surf, (200, self.height - 48 + i * 20))

                    spell_surf = self.font_s.render(text, True, color)
                    self.screen.blit(spell_surf, (205, self.height - 45 + i * 20))
                
                # Target selection
                selected_spell = self.available_spells[self.selected_spell_idx]
                if selected_spell['target'] == 'other':
                    target_title = self.font_m.render("Target (Left/Right):", True, self.COLOR_TEXT)
                    self.screen.blit(target_title, (self.width - 250, self.height - 80))
                    
                    target_options = [p for p in self.players if p['id'] != 0]
                    if target_options:
                        target = target_options[self.selected_target_idx % len(target_options)]
                        target_text = f"Player {target['id'] + 1}"
                        target_surf = self.font_m.render(target_text, True, target['color'])
                        self.screen.blit(target_surf, (self.width - 120, self.height - 80))

            else:
                text_surf = self.font_s.render("No spells available...", True, self.COLOR_TEXT_DIM)
                self.screen.blit(text_surf, (205, self.height - 45))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(overlay, (0, 0))
            
            if self.winner != -1:
                win_text = f"Player {self.winner + 1} Wins!"
                color = self.PLAYER_COLORS[self.winner]
            else:
                win_text = "Game Over"
                color = self.COLOR_TEXT
                
            text_surf = self.font_l.render(win_text, True, color)
            text_rect = text_surf.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(text_surf, text_rect)