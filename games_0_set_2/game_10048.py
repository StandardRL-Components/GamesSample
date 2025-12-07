import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:43:09.078945
# Source Brief: brief_00048.md
# Brief Index: 48
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Classes ---

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, life, size_range=(3, 8), speed_range=(1, 3)):
        self.x = x
        self.y = y
        self.vx = random.uniform(-speed_range[1], speed_range[1])
        self.vy = random.uniform(-speed_range[1], speed_range[1])
        self.life = life
        self.max_life = life
        self.color = color
        self.size = random.uniform(size_range[0], size_range[1])

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.size = max(0, self.size * 0.95)

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (int(self.size), int(self.size)), int(self.size))
            surface.blit(temp_surf, (int(self.x - self.size), int(self.y - self.size)), special_flags=pygame.BLEND_RGBA_ADD)

class Automaton:
    """Base class for all automatons."""
    COST = 1
    MAX_HP = 10
    ATTACK = 1
    RANGE = 1
    NAME = "BASE"
    SYMBOL = "?"
    COLOR = (128, 128, 128)

    def __init__(self, grid_pos, owner, difficulty_bonus=0):
        self.grid_pos = grid_pos
        self.pixel_pos = [grid_pos[0] * 40 + 20, grid_pos[1] * 40 + 20]
        self.target_pixel_pos = list(self.pixel_pos)
        self.owner = owner
        self.max_hp = self.MAX_HP
        self.hp = self.max_hp
        self.attack = self.ATTACK + (difficulty_bonus if owner == 'opponent' else 0)
        self.range = self.RANGE
        self.actions_this_turn = 1
        self.is_dead = False
        self.unique_id = random.randint(0, 1_000_000)

    def update_render_pos(self):
        # Interpolate movement
        self.pixel_pos[0] += (self.target_pixel_pos[0] - self.pixel_pos[0]) * 0.2
        self.pixel_pos[1] += (self.target_pixel_pos[1] - self.pixel_pos[1]) * 0.2

    def set_target_grid_pos(self, grid_pos):
        self.grid_pos = grid_pos
        self.target_pixel_pos = [grid_pos[0] * 40 + 20, grid_pos[1] * 40 + 20]

    def take_damage(self, amount):
        self.hp -= amount
        if self.hp <= 0:
            self.hp = 0
            self.is_dead = True
        return self.is_dead

    def draw_health_bar(self, surface):
        bar_width = 30
        bar_height = 4
        x, y = self.pixel_pos[0] - bar_width / 2, self.pixel_pos[1] - 25
        
        hp_ratio = self.hp / self.max_hp
        
        pygame.draw.rect(surface, (50, 50, 50), (x, y, bar_width, bar_height))
        pygame.draw.rect(surface, (0, 255, 0), (x, y, bar_width * hp_ratio, bar_height))

    def draw(self, surface):
        self.update_render_pos()
        x, y = int(self.pixel_pos[0]), int(self.pixel_pos[1])
        
        # Draw base
        pygame.gfxdraw.filled_circle(surface, x, y, 15, self.COLOR)
        pygame.gfxdraw.aacircle(surface, x, y, 15, (255, 255, 255))
        
        # Draw symbol
        font = pygame.font.SysFont("monospace", 20, bold=True)
        text = font.render(self.SYMBOL, True, (0, 0, 0))
        text_rect = text.get_rect(center=(x, y))
        surface.blit(text, text_rect)

        self.draw_health_bar(surface)

    def find_target(self, automatons, opponent_has_units):
        # Abstract method
        return None, None

    def perform_action(self, board, automatons, particles):
        # Abstract method
        return 0, False # damage_dealt, target_destroyed

class Pawn(Automaton):
    COST = 1
    MAX_HP = 15
    ATTACK = 2
    RANGE = 1.5 # Adjacent including diagonals
    NAME = "Pawn"
    SYMBOL = "P"

    def __init__(self, grid_pos, owner, difficulty_bonus=0):
        super().__init__(grid_pos, owner, difficulty_bonus)
        self.COLOR = (0, 150, 255) if owner == 'player' else (255, 100, 0)

    def find_target(self, automatons, opponent_has_units):
        targets = [a for a in automatons if a.owner != self.owner and not a.is_dead]
        if not targets:
            return None, 'opponent' # Target opponent player

        # Find closest target
        closest_target = min(targets, key=lambda t: math.hypot(t.grid_pos[0] - self.grid_pos[0], t.grid_pos[1] - self.grid_pos[1]))
        return closest_target, 'automaton'

    def perform_action(self, board, automatons, particles):
        opponent_has_units = any(a.owner != self.owner and not a.is_dead for a in automatons)
        target_automaton, target_type = self.find_target(automatons, opponent_has_units)
        
        if target_type == 'automaton' and target_automaton:
            dist = math.hypot(target_automaton.grid_pos[0] - self.grid_pos[0], target_automaton.grid_pos[1] - self.grid_pos[1])
            if dist <= self.RANGE:
                # Attack
                # SFX: Melee hit
                particles.extend([Particle(target_automaton.pixel_pos[0], target_automaton.pixel_pos[1], (255, 255, 100), 20) for _ in range(10)])
                destroyed = target_automaton.take_damage(self.attack)
                return self.attack, destroyed
            else:
                # Move towards target
                dx = target_automaton.grid_pos[0] - self.grid_pos[0]
                dy = target_automaton.grid_pos[1] - self.grid_pos[1]
                
                new_x, new_y = self.grid_pos
                if abs(dx) > abs(dy):
                    new_x += int(np.sign(dx))
                else:
                    new_y += int(np.sign(dy))

                if 0 <= new_x < 10 and 0 <= new_y < 10 and board[new_y][new_x] is None:
                    # SFX: Metal footstep
                    board[self.grid_pos[1]][self.grid_pos[0]] = None
                    self.set_target_grid_pos((new_x, new_y))
                    board[new_y][new_x] = self
        elif target_type == 'opponent':
            # Move towards opponent's side
            move_dir = -1 if self.owner == 'player' else 1
            new_y = self.grid_pos[1] + move_dir
            if 0 <= new_y < 10 and board[new_y][self.grid_pos[0]] is None:
                board[self.grid_pos[1]][self.grid_pos[0]] = None
                self.set_target_grid_pos((self.grid_pos[0], new_y))
                board[new_y][self.grid_pos[0]] = self
            elif self.grid_pos[1] == 0 or self.grid_pos[1] == 9:
                # Attack opponent player
                # SFX: Player damage taken
                particles.extend([Particle(self.pixel_pos[0], self.pixel_pos[1] + (move_dir * -20), (255, 0, 0), 20) for _ in range(10)])
                return self.attack, False

        return 0, False

class Rook(Automaton):
    COST = 2
    MAX_HP = 10
    ATTACK = 3
    RANGE = 4
    NAME = "Rook"
    SYMBOL = "R"
    
    def __init__(self, grid_pos, owner, difficulty_bonus=0):
        super().__init__(grid_pos, owner, difficulty_bonus)
        self.COLOR = (50, 180, 255) if owner == 'player' else (255, 130, 50)

    def find_target(self, automatons, opponent_has_units):
        targets = [a for a in automatons if a.owner != self.owner and not a.is_dead]
        
        # Prioritize targets in straight lines
        for target in sorted(targets, key=lambda t: math.hypot(t.grid_pos[0] - self.grid_pos[0], t.grid_pos[1] - self.grid_pos[1])):
            dx = target.grid_pos[0] - self.grid_pos[0]
            dy = target.grid_pos[1] - self.grid_pos[1]
            dist = math.hypot(dx, dy)
            if (dx == 0 or dy == 0) and dist <= self.RANGE:
                return target, 'automaton'
        
        if not opponent_has_units:
             return None, 'opponent'
        return None, None # Cannot find a valid target
    
    def perform_action(self, board, automatons, particles):
        opponent_has_units = any(a.owner != self.owner and not a.is_dead for a in automatons)
        target_automaton, target_type = self.find_target(automatons, opponent_has_units)
        
        if target_automaton:
            # SFX: Laser charge and fire
            particles.append(Projectile(self.pixel_pos, target_automaton.pixel_pos, (255, 255, 0), self.attack, target_automaton))
            return 0, False # Damage is handled by projectile
        elif target_type == 'opponent':
            if (self.owner == 'player' and self.grid_pos[1] <= self.RANGE) or \
               (self.owner == 'opponent' and 9 - self.grid_pos[1] <= self.RANGE):
                # SFX: Laser fire
                target_y = 0 if self.owner == 'player' else 399
                particles.append(Projectile(self.pixel_pos, [self.pixel_pos[0], target_y], (255, 0, 0), self.attack, target_type))
                return 0, False
        return 0, False

class Chronomancer(Automaton):
    COST = 3
    MAX_HP = 8
    ATTACK = 0
    RANGE = 2.5 # Area of effect for buff
    NAME = "Chronomancer"
    SYMBOL = "C"

    def __init__(self, grid_pos, owner, difficulty_bonus=0):
        super().__init__(grid_pos, owner, difficulty_bonus)
        self.COLOR = (150, 220, 255) if owner == 'player' else (255, 180, 150)
        self.actions_this_turn = 0 # Doesn't act itself

    def apply_buff(self, automatons, particles):
        # SFX: Time warp sound
        friendlies = [a for a in automatons if a.owner == self.owner and not a.is_dead and a.unique_id != self.unique_id]
        if not friendlies: return

        closest_friendly = min(friendlies, key=lambda t: math.hypot(t.grid_pos[0] - self.grid_pos[0], t.grid_pos[1] - self.grid_pos[1]))
        
        dist = math.hypot(closest_friendly.grid_pos[0] - self.grid_pos[0], closest_friendly.grid_pos[1] - self.grid_pos[1])
        if dist <= self.RANGE:
            closest_friendly.actions_this_turn += 1
            particles.extend([Particle(self.pixel_pos[0], self.pixel_pos[1], (255, 255, 0), 30, speed_range=(0,1)) for _ in range(15)])
            particles.extend([Particle(closest_friendly.pixel_pos[0], closest_friendly.pixel_pos[1], (255, 255, 0), 30, speed_range=(0,1)) for _ in range(15)])

    def perform_action(self, board, automatons, particles):
        return 0, False # Does nothing during its action phase

class Projectile(Particle):
    def __init__(self, start_pos, end_pos, color, damage, target):
        super().__init__(start_pos[0], start_pos[1], color, 100)
        self.start_pos = list(start_pos)
        self.end_pos = list(end_pos)
        self.damage = damage
        self.target = target
        self.speed = 15
        
        dist = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
        if dist > 0:
            self.vx = (end_pos[0] - start_pos[0]) / dist * self.speed
            self.vy = (end_pos[1] - start_pos[1]) / dist * self.speed
        else:
            self.vx = self.vy = 0
        self.life = int(dist / self.speed) if self.speed > 0 else 1

    def update(self):
        super().update()
        if self.life <= 1: # Hit
            self.life = 0
            if isinstance(self.target, Automaton):
                # SFX: Projectile impact
                destroyed = self.target.take_damage(self.damage)
                return self.damage, destroyed, self.target.owner
            elif self.target == 'opponent' or self.target == 'player':
                return self.damage, False, self.target
        return 0, False, None

    def draw(self, surface):
        if self.life > 0:
            pygame.draw.line(surface, self.color, (self.x, self.y), (self.x - self.vx, self.y - self.vy), 4)

# --- Main Environment Class ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A turn-based strategy game where you deploy clockwork automatons to battle your opponent. "
        "Manage your energy, choose your units wisely, and destroy the enemy's base."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press shift to cycle through available automatons. "
        "Press space to place a unit or end your turn."
    )
    auto_advance = False

    # --- Colors and Fonts ---
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (40, 50, 70)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_OPPONENT = (255, 100, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_CURSOR = (255, 255, 0)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        self.font_s = pygame.font.SysFont("Consolas", 14)
        self.font_m = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Persistent state
        self.wins = 0
        self.automaton_blueprints = {
            'player': [Pawn],
            'opponent': [Pawn]
        }
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_state = "PLAYER_TURN" # PLAYER_TURN, BATTLE_RESOLUTION, ENEMY_TURN
        self.last_reward = 0

        self.player_health = 100
        self.opponent_health = 100
        self.player_energy = 3
        self.opponent_energy = 3

        self.board = [[None for _ in range(10)] for _ in range(10)]
        self.automatons = []
        self.particles = []
        self.battle_queue = []

        self.cursor_pos = [4, 7]
        self.selected_automaton_idx = 0
        
        self.last_space_held = 0
        self.last_shift_held = 0
        
        # Opponent difficulty scaling
        self.difficulty_bonus = self.wins // 5

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        if self.game_state == "PLAYER_TURN":
            reward += self._handle_player_input(action)
        elif self.game_state == "BATTLE_RESOLUTION":
            reward += self._update_battle_resolution()
        elif self.game_state == "ENEMY_TURN":
            self._run_enemy_turn()
            self.game_state = "BATTLE_RESOLUTION"

        self.last_reward = reward
        self.score += reward
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.player_health <= 0:
                reward -= 100
            elif self.opponent_health <= 0:
                reward += 100
                self.wins += 1
                self._check_unlocks()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_input(self, action):
        movement, space_held, shift_held = action
        reward = -0.01 # Small penalty for taking time
        
        # --- Handle Button Presses (Rising Edge Detection) ---
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Movement ---
        if movement != 0: # 0 is no-op
            dx, dy = 0, 0
            if movement == 1: dy = -1 # Up
            elif movement == 2: dy = 1  # Down
            elif movement == 3: dx = -1 # Left
            elif movement == 4: dx = 1  # Right
            
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, 9)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 5, 9) # Player side

        # --- Cycle Automaton Selection ---
        if shift_pressed:
            # SFX: UI click
            self.selected_automaton_idx = (self.selected_automaton_idx + 1) % len(self.automaton_blueprints['player'])
        
        # --- Place Automaton / End Turn ---
        if space_pressed:
            x, y = self.cursor_pos
            can_place = self.board[y][x] is None
            
            selected_class = self.automaton_blueprints['player'][self.selected_automaton_idx]
            has_energy = self.player_energy >= selected_class.COST

            if can_place and has_energy:
                # SFX: Unit placement
                self.player_energy -= selected_class.COST
                new_automaton = selected_class((x, y), 'player')
                self.automatons.append(new_automaton)
                self.board[y][x] = new_automaton
                self._start_battle_phase()
            else:
                # Invalid placement or not enough energy, interpret as ending turn
                self._start_battle_phase()
        
        return reward

    def _start_battle_phase(self):
        # SFX: Turn start chime
        # Chronomancers apply buffs first
        for automaton in self.automatons:
            if isinstance(automaton, Chronomancer):
                automaton.apply_buff(self.automatons, self.particles)

        # Build battle queue
        self.battle_queue = []
        for automaton in sorted(self.automatons, key=lambda a: a.grid_pos[1]):
            for _ in range(automaton.actions_this_turn):
                self.battle_queue.append(automaton)
        
        self.game_state = "BATTLE_RESOLUTION"

    def _update_battle_resolution(self):
        reward = 0
        
        # Update particles and projectiles
        new_particles = []
        for p in self.particles:
            if isinstance(p, Projectile):
                dmg, destroyed, owner = p.update()
                if dmg > 0:
                    if owner == 'player':
                        self.player_health -= dmg
                        reward -= 0.1 * dmg
                    elif owner == 'opponent':
                        self.opponent_health -= dmg
                        reward += 0.1 * dmg
                    if destroyed:
                        reward += 1.0 if owner == 'player' else -1.0 # Reward for destroying opponent, penalty for losing own
                if p.life > 0:
                    new_particles.append(p)
            else: # Standard particle
                p.update()
                if p.life > 0:
                    new_particles.append(p)
        self.particles = new_particles

        # If animations are playing, wait for them
        if any(isinstance(p, Projectile) for p in self.particles):
            return reward

        if not self.battle_queue:
            # End of battle resolution
            current_turn_owner = self.automatons[0].owner if self.automatons else 'player'
            if current_turn_owner == 'player':
                self.game_state = "ENEMY_TURN"
            else:
                self.player_energy = min(10, self.player_energy + 1)
                self.game_state = "PLAYER_TURN"
            
            # Reset action counts
            for a in self.automatons:
                a.actions_this_turn = 1
            return reward

        automaton = self.battle_queue.pop(0)
        if automaton.is_dead:
            return reward

        dmg_dealt, target_destroyed = automaton.perform_action(self.board, self.automatons, self.particles)

        if dmg_dealt > 0:
            if automaton.owner == 'player':
                self.opponent_health -= dmg_dealt
                reward += 0.1 * dmg_dealt
            else:
                self.player_health -= dmg_dealt
                reward -= 0.1 * dmg_dealt
        
        if target_destroyed:
            reward += 1.0 if automaton.owner == 'player' else -1.0
        
        # Clean up dead automatons
        dead_automatons = [a for a in self.automatons if a.is_dead]
        for dead in dead_automatons:
            # SFX: Explosion
            self.particles.extend([Particle(dead.pixel_pos[0], dead.pixel_pos[1], (200, 200, 200), 40, size_range=(5,15)) for _ in range(30)])
            if self.board[dead.grid_pos[1]][dead.grid_pos[0]] == dead:
                self.board[dead.grid_pos[1]][dead.grid_pos[0]] = None
        self.automatons = [a for a in self.automatons if not a.is_dead]

        return reward

    def _run_enemy_turn(self):
        self.opponent_energy = min(10, self.opponent_energy + 1)
        
        # AI: Try to place the most expensive unit it can afford in a random valid spot
        available_blueprints = sorted(self.automaton_blueprints['opponent'], key=lambda b: b.COST, reverse=True)
        
        placed = False
        for blueprint in available_blueprints:
            if self.opponent_energy >= blueprint.COST:
                possible_positions = []
                for y in range(0, 5): # Opponent's side
                    for x in range(10):
                        if self.board[y][x] is None:
                            possible_positions.append((x, y))
                
                if possible_positions:
                    pos = random.choice(possible_positions)
                    new_automaton = blueprint(pos, 'opponent', self.difficulty_bonus)
                    self.automatons.append(new_automaton)
                    self.board[pos[1]][pos[0]] = new_automaton
                    self.opponent_energy -= blueprint.COST
                    placed = True
                    break
        
        # Always start battle phase, even if AI did nothing
        self._start_battle_phase()

    def _check_unlocks(self):
        if self.wins == 1 and Rook not in self.automaton_blueprints['player']:
            self.automaton_blueprints['player'].append(Rook)
            self.automaton_blueprints['opponent'].append(Rook)
            self.score += 5 # Unlock reward
        if self.wins == 3 and Chronomancer not in self.automaton_blueprints['player']:
            self.automaton_blueprints['player'].append(Chronomancer)
            self.automaton_blueprints['opponent'].append(Chronomancer)
            self.score += 5 # Unlock reward

    def _check_termination(self):
        return self.player_health <= 0 or self.opponent_health <= 0 or self.steps >= 1000

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
            "player_health": self.player_health,
            "opponent_health": self.opponent_health,
            "wins": self.wins,
        }

    def _render_game(self):
        # Draw grid
        for i in range(11):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i * 40, 0), (i * 40, 400), 1)
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i * 40), (400, i * 40), 1)
        pygame.draw.line(self.screen, (100, 100, 120), (0, 200), (400, 200), 2) # Center line

        # Draw automatons
        for automaton in self.automatons:
            automaton.draw(self.screen)
        
        # Draw player cursor
        if self.game_state == "PLAYER_TURN":
            x, y = self.cursor_pos
            px, py = x * 40, y * 40
            
            # Check validity for color
            selected_class = self.automaton_blueprints['player'][self.selected_automaton_idx]
            is_valid = self.board[y][x] is None and self.player_energy >= selected_class.COST
            cursor_color = self.COLOR_CURSOR if is_valid else (255, 0, 0)
            
            rect = pygame.Rect(px, py, 40, 40)
            # Glow effect
            for i in range(4):
                glow_rect = rect.inflate(i * 4, i * 4)
                alpha = 100 - i * 25
                shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(shape_surf, (*cursor_color, alpha), (0, 0, *glow_rect.size), border_radius=5)
                self.screen.blit(shape_surf, glow_rect.topleft)
            pygame.draw.rect(self.screen, cursor_color, rect, 2, border_radius=5)

        # Draw particles and projectiles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        ui_x = 420
        # --- Player UI ---
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (ui_x - 10, 210, 220, 180), 2, 5)
        self.screen.blit(self.font_l.render("PLAYER", True, self.COLOR_PLAYER), (ui_x, 220))
        self.screen.blit(self.font_m.render(f"HP: {int(self.player_health)}/100", True, self.COLOR_TEXT), (ui_x, 250))
        self.screen.blit(self.font_m.render(f"Energy: {self.player_energy}", True, self.COLOR_TEXT), (ui_x, 270))
        
        # --- Opponent UI ---
        pygame.draw.rect(self.screen, self.COLOR_OPPONENT, (ui_x - 10, 10, 220, 100), 2, 5)
        self.screen.blit(self.font_l.render("OPPONENT", True, self.COLOR_OPPONENT), (ui_x, 20))
        self.screen.blit(self.font_m.render(f"HP: {int(self.opponent_health)}/100", True, self.COLOR_TEXT), (ui_x, 50))
        self.screen.blit(self.font_m.render(f"Energy: {self.opponent_energy}", True, self.COLOR_TEXT), (ui_x, 70))
        
        # --- Game State UI ---
        self.screen.blit(self.font_m.render(f"Turn: {self.game_state}", True, self.COLOR_TEXT), (ui_x, 150))
        self.screen.blit(self.font_m.render(f"Wins: {self.wins}", True, self.COLOR_TEXT), (ui_x, 170))

        # --- Selected Automaton UI ---
        if self.game_state == "PLAYER_TURN":
            selected_class = self.automaton_blueprints['player'][self.selected_automaton_idx]
            self.screen.blit(self.font_m.render("Selected:", True, self.COLOR_TEXT), (ui_x, 300))
            name_text = self.font_l.render(selected_class.NAME, True, self.COLOR_CURSOR)
            self.screen.blit(name_text, (ui_x, 320))
            self.screen.blit(self.font_s.render(f"Cost: {selected_class.COST}, HP: {selected_class.MAX_HP}, ATK: {selected_class.ATTACK}", True, self.COLOR_TEXT), (ui_x, 350))
        
        # --- Game Over Screen ---
        if self.game_over:
            s = pygame.Surface((640, 400), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            msg = "VICTORY" if self.player_health > 0 else "DEFEAT"
            color = (0, 255, 0) if msg == "VICTORY" else (255, 0, 0)
            text = self.font_l.render(msg, True, color)
            text_rect = text.get_rect(center=(320, 200))
            self.screen.blit(text, text_rect)

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # For this to work, you must comment out the os.environ line at the top of the file
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    pygame.display.set_caption("Clockwork Tactics")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # no-op, no-space, no-shift
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
        
        # Map keyboard to MultiDiscrete action
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    pygame.quit()