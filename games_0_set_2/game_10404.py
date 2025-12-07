import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:20:06.604040
# Source Brief: brief_00404.md
# Brief Index: 404
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import copy

class Unit:
    """Represents a player or enemy unit."""
    _id_counter = 0
    def __init__(self, pos, hp, attack, team, type_name="Grunt"):
        self.id = Unit._id_counter
        Unit._id_counter += 1
        self.pos = list(pos)
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.team = team # 'player' or 'enemy'
        self.type_name = type_name
        
        # For animation
        self.render_pos = [p * 40 + 20 for p in self.pos]
        self.is_new = True

    def to_dict(self):
        return {
            "id": self.id, "pos": self.pos, "hp": self.hp, "max_hp": self.max_hp,
            "attack": self.attack, "team": self.team, "type_name": self.type_name
        }

    @classmethod
    def from_dict(cls, data):
        unit = cls(data['pos'], data['max_hp'], data['attack'], data['team'], data['type_name'])
        unit.id = data['id']
        unit.hp = data['hp']
        return unit

class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, x, y, color, speed, angle, life):
        self.x = x
        self.y = y
        self.color = color
        self.vx = math.cos(angle) * speed * (0.5 + random.random())
        self.vy = math.sin(angle) * speed * (0.5 + random.random())
        self.life = life
        self.max_life = life

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.98
        self.vy *= 0.98
        self.life -= 1

class FloatingText:
    """Represents floating text for damage numbers or status effects."""
    def __init__(self, text, pos, color, font):
        self.text = text
        self.pos = list(pos)
        self.color = color
        self.font = font
        self.surface = font.render(text, True, color)
        self.life = 60 # 2 seconds at 30fps
        self.y_vel = -1

    def update(self):
        self.pos[1] += self.y_vel
        self.y_vel += 0.05
        self.life -= 1

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A tactical turn-based game where you command units and manipulate time. Deploy units into the past to alter the timeline, but beware of creating paradoxes."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press [SPACE] to select/deselect units and portals. "
        "With a unit and portal selected, press [SHIFT] to deploy."
    )
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_W, GRID_H = 16, 10
    TILE_SIZE = 40
    MAX_STEPS = 1000
    HISTORY_LENGTH = 10
    PORTAL_LOOKBACK = 3

    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 60)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (100, 200, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 120, 120)
    COLOR_PORTAL = (50, 255, 150)
    COLOR_PORTAL_GLOW = (150, 255, 200)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_PARADOX_BAR = (255, 180, 0)
    COLOR_HEALTH_GREEN = (0, 200, 0)
    COLOR_HEALTH_YELLOW = (200, 200, 0)
    COLOR_HEALTH_RED = (200, 0, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_floating = pygame.font.SysFont("Arial", 16, bold=True)
        
        self.last_space_held = False
        self.last_shift_held = False

        # self.reset() is called by the wrapper
        # self.validate_implementation() is for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        Unit._id_counter = 0

        self.steps = 0
        self.turn = 0
        self.score = 0
        self.game_over = False
        self.paradox_meter = 0.0

        self.player_units = []
        self.enemy_units = []
        
        self.portals = [
            {'pos': (2, 2), 'id': 0}, {'pos': (self.GRID_W - 3, 2), 'id': 1},
            {'pos': (2, self.GRID_H - 3), 'id': 2}, {'pos': (self.GRID_W - 3, self.GRID_H - 3), 'id': 3}
        ]
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_unit_id = None
        self.selected_portal_id = None
        
        self.particles = []
        self.floating_texts = []
        
        self.history = collections.deque(maxlen=self.HISTORY_LENGTH)
        self._spawn_initial_units()
        self._save_state_to_history()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        
        self._handle_input(movement, space_pressed, shift_pressed)
        
        turn_processed = False
        if shift_pressed and self.selected_unit_id is not None and self.selected_portal_id is not None:
            # Main game action: Deploy unit through portal
            deployment_reward, paradox_change = self._perform_deployment()
            reward += deployment_reward
            self.paradox_meter = min(100, self.paradox_meter + paradox_change)
            
            enemy_turn_reward = self._run_enemy_turn()
            reward += enemy_turn_reward
            
            self._cleanup_dead_units()
            self._spawn_enemies()
            
            self.turn += 1
            self._save_state_to_history()
            
            self.selected_unit_id = None
            self.selected_portal_id = None
            turn_processed = True
            # sfx: positive confirmation sound

        self._update_animations()
        
        self.steps += 1
        self.score += reward
        
        if self.paradox_meter >= 100:
            reward -= 100
            terminated = True
            # sfx: failure sound
        elif not any(u for u in self.enemy_units if u.team == 'enemy'):
             if self.turn > 5: # Win only after at least a few turns
                reward += 100
                terminated = True
                # sfx: victory fanfare
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            
        self.game_over = terminated
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        if movement == 2: self.cursor_pos[1] += 1 # Down
        if movement == 3: self.cursor_pos[0] -= 1 # Left
        if movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        if not space_pressed: return

        # --- Selection Logic (Spacebar) ---
        cursor_tuple = tuple(self.cursor_pos)
        
        # Try to select a player unit
        unit_at_cursor = next((u for u in self.player_units if tuple(u.pos) == cursor_tuple), None)
        if unit_at_cursor:
            if self.selected_unit_id == unit_at_cursor.id:
                self.selected_unit_id = None # Deselect
                # sfx: deselect sound
            else:
                self.selected_unit_id = unit_at_cursor.id
                self.selected_portal_id = None # Clear portal selection
                # sfx: select sound
            return

        # Try to select a portal (only if a unit is already selected)
        if self.selected_unit_id is not None:
            portal_at_cursor = next((p for p in self.portals if p['pos'] == cursor_tuple), None)
            if portal_at_cursor:
                if self.selected_portal_id == portal_at_cursor['id']:
                    self.selected_portal_id = None # Deselect
                    # sfx: deselect sound
                else:
                    self.selected_portal_id = portal_at_cursor['id']
                    # sfx: select sound

    def _perform_deployment(self):
        unit_to_deploy = next((u for u in self.player_units if u.id == self.selected_unit_id), None)
        portal = next((p for p in self.portals if p['id'] == self.selected_portal_id), None)
        
        if not unit_to_deploy or not portal: return 0, 0
        if self.turn < self.PORTAL_LOOKBACK: return 0, 0 # Cannot deploy to non-existent past

        # sfx: time travel whoosh
        self._create_particle_burst(
            (unit_to_deploy.pos[0] + 0.5) * self.TILE_SIZE,
            (unit_to_deploy.pos[1] + 0.5) * self.TILE_SIZE,
            self.COLOR_PLAYER, 30
        )

        original_state = self.history[-1]
        
        # Modify the past
        past_turn_idx = -1 - self.PORTAL_LOOKBACK
        past_state = self.history[past_turn_idx]
        
        modified_history = list(self.history)
        
        # Create a new unit in the past, remove from present
        new_unit_in_past = copy.deepcopy(unit_to_deploy)
        new_unit_in_past.pos = list(portal['pos']) # Appears at portal location
        modified_history[past_turn_idx]['player'].append(new_unit_in_past)

        current_player_units = [u for u in modified_history[-1]['player'] if u.id != unit_to_deploy.id]
        
        # Re-simulate history
        resimulated_state = self._re_simulate(past_turn_idx, modified_history)
        
        # Calculate paradox
        paradox_change = self._calculate_paradox(original_state, resimulated_state)

        # Update current state to the new timeline
        self.player_units = current_player_units + resimulated_state['player']
        self.enemy_units = resimulated_state['enemy']
        
        self._create_particle_burst(
            (portal['pos'][0] + 0.5) * self.TILE_SIZE,
            (portal['pos'][1] + 0.5) * self.TILE_SIZE,
            self.COLOR_PORTAL, 30
        )

        return 0, paradox_change # Reward for deployment is neutral

    def _re_simulate(self, start_idx, history_list):
        temp_history = copy.deepcopy(history_list)
        
        for i in range(len(temp_history) + start_idx, len(temp_history) -1):
            current_player = temp_history[i]['player']
            current_enemy = temp_history[i]['enemy']
            
            # Simple combat simulation: all enemies attack closest player
            for enemy in current_enemy:
                if not current_player: break
                target = min(current_player, key=lambda p: self._dist(p.pos, enemy.pos))
                target.hp -= enemy.attack

            # All players attack closest enemy
            for player in current_player:
                if not current_enemy: break
                target = min(current_enemy, key=lambda e: self._dist(e.pos, player.pos))
                target.hp -= player.attack

            # Update next state
            temp_history[i+1]['player'] = [p for p in current_player if p.hp > 0]
            temp_history[i+1]['enemy'] = [e for e in current_enemy if e.hp > 0]

        return temp_history[-1]

    def _calculate_paradox(self, original_state, new_state):
        paradox = 0
        
        orig_units = {u.id: u for u in original_state['player'] + original_state['enemy']}
        new_units = {u.id: u for u in new_state['player'] + new_state['enemy']}
        
        all_ids = set(orig_units.keys()) | set(new_units.keys())
        
        for uid in all_ids:
            u_orig = orig_units.get(uid)
            u_new = new_units.get(uid)
            
            if u_orig and not u_new: paradox += 5 # Unit was erased
            elif not u_orig and u_new: paradox += 5 # Unit created from nothing
            elif u_orig and u_new:
                paradox += abs(u_orig.hp - u_new.hp) * 0.5 # Health changed
        
        if paradox > 0:
            self._add_floating_text(f"PARADOX +{paradox:.1f}", [self.WIDTH/2, 50], self.COLOR_PARADOX_BAR)
            # sfx: paradox alert
        return paradox

    def _run_enemy_turn(self):
        reward = 0
        for enemy in self.enemy_units:
            if not self.player_units: break
            target = min(self.player_units, key=lambda p: self._dist(p.pos, enemy.pos))
            damage = enemy.attack
            
            self._add_floating_text(f"-{damage}", self._grid_to_pixel(target.pos), self.COLOR_ENEMY)
            self._create_particle_burst(self._grid_to_pixel(target.pos)[0], self._grid_to_pixel(target.pos)[1], self.COLOR_ENEMY, 10, speed=2)
            
            target.hp -= damage
            reward -= 0.1 # Penalty for player taking damage
            
            if target.hp <= 0:
                reward -= 5 # Penalty for losing a unit
                # sfx: player unit destroyed
        return reward

    def _cleanup_dead_units(self):
        reward = 0
        dead_player_units = [u for u in self.player_units if u.hp <= 0]
        dead_enemy_units = [u for u in self.enemy_units if u.hp <= 0]
        
        for u in dead_player_units + dead_enemy_units:
            color = self.COLOR_PLAYER if u.team == 'player' else self.COLOR_ENEMY
            self._create_particle_burst(self._grid_to_pixel(u.pos)[0], self._grid_to_pixel(u.pos)[1], color, 50, speed=4)
        
        if dead_enemy_units: reward += 5 * len(dead_enemy_units)
        
        self.player_units = [u for u in self.player_units if u.hp > 0]
        self.enemy_units = [u for u in self.enemy_units if u.hp > 0]
        return reward

    def _spawn_initial_units(self):
        self.player_units.append(Unit(pos=(1, self.GRID_H // 2), hp=20, attack=5, team='player', type_name="Marine"))
        self.player_units.append(Unit(pos=(1, self.GRID_H // 2 - 1), hp=15, attack=8, team='player', type_name="Ranger"))

    def _spawn_enemies(self):
        base_spawn_chance = 0.3
        difficulty_scaling = 1 + (self.turn // 5) * 0.2
        
        if self.np_random.random() < base_spawn_chance * difficulty_scaling:
            spawn_x = self.GRID_W - 2
            spawn_y = self.np_random.integers(0, self.GRID_H)
            
            base_hp = 10 + (self.turn // 10)
            base_attack = 3 + (self.turn // 8)
            
            self.enemy_units.append(Unit(pos=(spawn_x, spawn_y), hp=base_hp, attack=base_attack, team='enemy'))
            # sfx: enemy spawn alert

    def _save_state_to_history(self):
        player_state = [u.to_dict() for u in self.player_units]
        enemy_state = [u.to_dict() for u in self.enemy_units]
        
        state_snapshot = {
            'player': [Unit.from_dict(d) for d in player_state],
            'enemy': [Unit.from_dict(d) for d in enemy_state]
        }
        self.history.append(state_snapshot)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw portals
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 * 3
        for p in self.portals:
            px, py = self._grid_to_pixel(p['pos'])
            is_selected = self.selected_portal_id == p['id']
            radius = int(self.TILE_SIZE * 0.4 + (pulse if not is_selected else 5))
            color = self.COLOR_CURSOR if is_selected else self.COLOR_PORTAL
            glow_color = self.COLOR_CURSOR if is_selected else self.COLOR_PORTAL_GLOW
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, glow_color)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, color)

        # Draw units
        for unit in self.player_units + self.enemy_units:
            target_px, target_py = self._grid_to_pixel(unit.pos)
            unit.render_pos[0] += (target_px - unit.render_pos[0]) * 0.2
            unit.render_pos[1] += (target_py - unit.render_pos[1]) * 0.2
            px, py = int(unit.render_pos[0]), int(unit.render_pos[1])
            
            is_selected = self.selected_unit_id == unit.id
            color = self.COLOR_PLAYER if unit.team == 'player' else self.COLOR_ENEMY
            glow_color = self.COLOR_PLAYER_GLOW if unit.team == 'player' else self.COLOR_ENEMY_GLOW
            
            if is_selected:
                pygame.gfxdraw.filled_circle(self.screen, px, py, 20, self.COLOR_CURSOR)
            
            # Draw unit shape
            if unit.team == 'player':
                points = [(px, py - 10), (px - 9, py + 7), (px + 9, py + 7)]
                pygame.gfxdraw.filled_trigon(self.screen, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], glow_color)
                pygame.gfxdraw.aatrigon(self.screen, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], color)
            else:
                size = 10
                pygame.draw.rect(self.screen, glow_color, (px - size, py - size, size*2, size*2))
                pygame.draw.rect(self.screen, color, (px - size, py - size, size*2, size*2), 2)

            # Draw health bar
            hp_ratio = max(0, unit.hp / unit.max_hp)
            bar_w, bar_h = 30, 5
            bar_x, bar_y = px - bar_w // 2, py - 25
            hp_color = self.COLOR_HEALTH_GREEN if hp_ratio > 0.6 else self.COLOR_HEALTH_YELLOW if hp_ratio > 0.3 else self.COLOR_HEALTH_RED
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, hp_color, (bar_x, bar_y, int(bar_w * hp_ratio), bar_h))
            
        # Draw cursor
        cursor_px, cursor_py = self._grid_to_pixel(self.cursor_pos)
        size = self.TILE_SIZE // 2
        alpha = 128 + 127 * math.sin(self.steps * 0.2)
        cursor_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surf, (*self.COLOR_CURSOR, alpha), (0,0,size*2,size*2), 2)
        self.screen.blit(cursor_surf, (cursor_px - size, cursor_py - size))
        
        # Draw particles & floating text
        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), int(p.life * 0.1), (*p.color, alpha))
        for t in self.floating_texts:
            alpha = int(255 * (t.life / 60))
            t.surface.set_alpha(alpha)
            self.screen.blit(t.surface, t.pos)
            
    def _render_ui(self):
        # Paradox Meter
        bar_width = self.WIDTH - 40
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_GRID, (20, 10, bar_width, bar_height))
        paradox_width = (self.paradox_meter / 100) * bar_width
        pygame.draw.rect(self.screen, self.COLOR_PARADOX_BAR, (20, 10, paradox_width, bar_height))
        if self.paradox_meter > 50: # Add jitter effect
            offset_x = random.randint(-2,2)
            offset_y = random.randint(-2,2)
            pygame.draw.rect(self.screen, self.COLOR_PARADOX_BAR, (20+offset_x, 10+offset_y, paradox_width, bar_height))

        ui_text = f"PARADOX: {self.paradox_meter:.1f}%"
        self._draw_text(ui_text, (self.WIDTH / 2, 20), self.font_ui, self.COLOR_UI_TEXT)
        self._draw_text(f"TURN: {self.turn}", (50, self.HEIGHT - 20), self.font_ui, self.COLOR_UI_TEXT)
        self._draw_text(f"SCORE: {self.score:.1f}", (self.WIDTH - 80, self.HEIGHT - 20), self.font_ui, self.COLOR_UI_TEXT)

        # Selection Info
        info_text = "ACTION: Move cursor [ARROWS], Select [SPACE], Deploy [SHIFT]"
        if self.selected_unit_id is not None:
            unit = next((u for u in self.player_units if u.id == self.selected_unit_id), None)
            if unit:
                info_text = f"SELECTED: {unit.type_name} (HP: {unit.hp}/{unit.max_hp}, ATK: {unit.attack}). "
                if self.selected_portal_id is not None:
                    info_text += f"TARGET: Portal {self.selected_portal_id}. Press [SHIFT] to deploy."
                else:
                    info_text += "Select a Portal [SPACE]."
        self._draw_text(info_text, (self.WIDTH/2, self.HEIGHT - 20), self.font_ui, self.COLOR_UI_TEXT)

    def _update_animations(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles: p.update()
        self.floating_texts = [t for t in self.floating_texts if t.life > 0]
        for t in self.floating_texts: t.update()

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "turn": self.turn, "paradox": self.paradox_meter}

    # --- Utility Methods ---
    def _grid_to_pixel(self, grid_pos):
        return int((grid_pos[0] + 0.5) * self.TILE_SIZE), int((grid_pos[1] + 0.5) * self.TILE_SIZE)
    def _dist(self, p1, p2): return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    def _draw_text(self, text, pos, font, color):
        surface = font.render(text, True, color)
        rect = surface.get_rect(center=pos)
        self.screen.blit(surface, rect)
    def _create_particle_burst(self, x, y, color, count, speed=3):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            life = random.randint(20, 40)
            self.particles.append(Particle(x, y, color, speed, angle, life))
    def _add_floating_text(self, text, pos, color):
        self.floating_texts.append(FloatingText(text, pos, color, self.font_floating))

    def validate_implementation(self):
        # This method is for developer convenience and can be removed.
        print("✓ Validating implementation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually.
    # It will create a window and render the game.
    os.environ['SDL_VIDEODRIVER'] = 'x11' # or 'windows', 'mac', etc.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Quantum Paradox")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Turn: {info['turn']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Paradox: {info['paradox']:.1f}%")

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            obs, info = env.reset()

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    pygame.quit()