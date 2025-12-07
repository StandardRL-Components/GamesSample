import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()


# Generated: 2025-08-26T10:56:31.673853
# Source Brief: brief_00765.md
# Brief Index: 765
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Classes ---

class Unit:
    def __init__(self, x, y, team, health, unit_id):
        self.pos = pygame.Vector2(x, y)
        self.team = team
        self.max_health = health
        self.health = health
        self.radius = 12
        self.attack_range = 100
        self.attack_cooldown = 0
        self.attack_rate = 30  # 1 attack per second at 30 FPS
        self.target = None
        self.id = unit_id
        self.is_overcharged = False
        self.overcharge_timer = 0
        self.move_speed = 1.5

    def find_target(self, enemy_units):
        closest_enemy = None
        min_dist = float('inf')
        for enemy in enemy_units:
            dist = self.pos.distance_to(enemy.pos)
            if dist < min_dist:
                min_dist = dist
                closest_enemy = enemy
        self.target = closest_enemy

    def update(self, enemy_units, reward_accumulator):
        # Update overcharge status
        if self.is_overcharged:
            self.overcharge_timer -= 1
            if self.overcharge_timer <= 0:
                self.is_overcharged = False

        # Cooldown management
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        # Targeting
        self.find_target(enemy_units)

        # AI: Attack or Move
        if self.target:
            dist_to_target = self.pos.distance_to(self.target.pos)
            if dist_to_target <= self.attack_range:
                # In range: Attack
                if self.attack_cooldown == 0:
                    attack_rate = self.attack_rate / 2 if self.is_overcharged else self.attack_rate
                    self.attack_cooldown = attack_rate
                    return self._attack(reward_accumulator)
            else:
                # Out of range: Move towards target
                direction = (self.target.pos - self.pos).normalize()
                self.pos += direction * self.move_speed
        elif self.team == 'enemy':
            # Enemy patrol behavior: move towards center
            center = pygame.Vector2(640 / 2, 400 / 2)
            if self.pos.distance_to(center) > 10:
                direction = (center - self.pos).normalize()
                self.pos += direction * self.move_speed * 0.5

        return None, None # No attack happened

    def _attack(self, reward_accumulator):
        if self.team == 'player':
            reward_accumulator['momentum_gain'] += 5
        # Returns a projectile start and end for rendering
        return self.pos, self.target.pos

    def take_damage(self, amount, reward_accumulator):
        self.health -= amount
        if self.team == 'enemy':
             reward_accumulator['damage_dealt'] += 1
        return self.health <= 0

class Sector:
    def __init__(self, rect, name):
        self.rect = rect
        self.name = name
        self.owner = 'neutral' # 'player', 'enemy', 'neutral'
        self.capture_progress = 0 # -100 (enemy) to 100 (player)
        self.color_map = {
            'player': (50, 50, 150, 100),
            'enemy': (150, 50, 50, 100),
            'neutral': (100, 100, 100, 50)
        }

    def update_owner(self, player_units, enemy_units, reward_accumulator):
        player_count = sum(1 for u in player_units if self.rect.collidepoint(u.pos))
        enemy_count = sum(1 for u in enemy_units if self.rect.collidepoint(u.pos))

        if player_count > enemy_count:
            self.capture_progress = min(100, self.capture_progress + 2)
        elif enemy_count > player_count:
            self.capture_progress = max(-100, self.capture_progress - 2)
        else: # Equal or zero units
            if self.capture_progress > 0:
                self.capture_progress -= 1
            elif self.capture_progress < 0:
                self.capture_progress += 1

        new_owner = self.owner
        if self.capture_progress >= 100:
            new_owner = 'player'
        elif self.capture_progress <= -100:
            new_owner = 'enemy'
        else:
            new_owner = 'neutral'

        if new_owner != self.owner and new_owner == 'player':
            reward_accumulator['sector_captured'] += 1
        
        self.owner = new_owner

    def draw(self, surface):
        color = self.color_map[self.owner]
        s = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        s.fill(color)
        surface.blit(s, self.rect.topleft)

class ComboCard:
    def __init__(self, name, cost):
        self.name = name
        self.cost = cost

# --- Main Environment Class ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A cyberpunk real-time strategy game. Deploy units to capture sectors, manage resources, "
        "and unleash powerful combo abilities to overwhelm your opponent."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to deploy units from different sides of the map. "
        "Press space to activate the first combo card and shift to activate the second."
    )
    auto_advance = True


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Clock
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        pygame.font.init()

        # Spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Colors
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_GRID = (30, 20, 50)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_ENEMY = (255, 50, 100)
        self.COLOR_MOMENTUM = (255, 200, 0)
        self.COLOR_COMBO = (180, 50, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_GREEN = (0, 200, 0)
        self.COLOR_HEALTH_RED = (200, 0, 0)

        # Fonts
        try:
            self.FONT_UI = pygame.font.SysFont("Consolas", 18)
            self.FONT_CARD = pygame.font.SysFont("Consolas", 14)
        except pygame.error:
            self.FONT_UI = pygame.font.SysFont(None, 24)
            self.FONT_CARD = pygame.font.SysFont(None, 20)
            
        # Game State (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.max_steps = 5000
        self.next_unit_id = 0

        self.player_units = []
        self.enemy_units = []
        self.sectors = []
        self.particles = []
        self.projectiles = []
        
        self.momentum = 0
        self.max_momentum = 100
        self.combo_cards = []
        self.combo_card_slots = 2
        
        self.enemy_spawn_timer = 0
        self.base_enemy_spawn_rate = 1.0 # units per second
        self.current_enemy_spawn_rate = self.base_enemy_spawn_rate
        self.base_enemy_health = 40
        self.current_enemy_health = self.base_enemy_health

        # Action state trackers
        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False

    def _get_next_unit_id(self):
        self.next_unit_id += 1
        return self.next_unit_id

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.next_unit_id = 0

        self.player_units.clear()
        self.enemy_units.clear()
        self.particles.clear()
        self.projectiles.clear()
        self.sectors.clear()

        # Initialize game elements
        self._initialize_sectors()
        self._spawn_unit('player', self.WIDTH / 4, self.HEIGHT / 2)
        self._spawn_unit('enemy', self.WIDTH * 3/4, self.HEIGHT / 2)

        # Reset progression and resources
        self.momentum = 25
        self.combo_cards = [None, None]
        self._generate_combo_card(0)
        self._generate_combo_card(1)
        
        self.current_enemy_spawn_rate = self.base_enemy_spawn_rate
        self.current_enemy_health = self.base_enemy_health
        self.enemy_spawn_timer = 0

        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False

        return self._get_observation(), self._get_info()

    def _initialize_sectors(self):
        w, h = self.WIDTH / 2, self.HEIGHT / 2
        self.sectors.append(Sector(pygame.Rect(0, 0, w, h), "Alpha"))
        self.sectors.append(Sector(pygame.Rect(w, 0, w, h), "Bravo"))
        self.sectors.append(Sector(pygame.Rect(0, h, w, h), "Charlie"))
        self.sectors.append(Sector(pygame.Rect(w, h, w, h), "Delta"))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward_accumulator = {
            'damage_dealt': 0,
            'enemy_destroyed': 0,
            'sector_captured': 0,
            'combo_used': 0,
            'momentum_gain': 0
        }

        self._handle_actions(action, reward_accumulator)
        self._update_game_state(reward_accumulator)
        
        reward = self._calculate_reward(reward_accumulator)
        self.score += reward
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if len(self.player_units) == 0:
                reward -= 100 # Loss
            elif all(s.owner == 'player' for s in self.sectors):
                reward += 100 # Win
        
        self.steps += 1
        truncated = self.steps >= self.max_steps

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_actions(self, action, reward_accumulator):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Action 0: Deploy unit
        if movement > 0:
            cost = 50
            if self.momentum >= cost:
                self.momentum -= cost
                if movement == 1: # Up
                    self._spawn_unit('player', random.randint(50, self.WIDTH - 50), 20)
                elif movement == 2: # Down
                    self._spawn_unit('player', random.randint(50, self.WIDTH - 50), self.HEIGHT - 20)
                elif movement == 3: # Left
                    self._spawn_unit('player', 20, random.randint(50, self.HEIGHT - 50))
                elif movement == 4: # Right
                    self._spawn_unit('player', self.WIDTH - 20, random.randint(50, self.HEIGHT - 50))
                # sfx: unit_deployed.wav
                self._create_particles(self.player_units[-1].pos, 20, self.COLOR_PLAYER, 1)

        # Action 1: Activate first combo card (on press)
        space_pressed = space_held and not self.space_pressed_last_frame
        if space_pressed and self.combo_cards[0] is not None:
            if self.momentum >= self.combo_cards[0].cost:
                self.momentum -= self.combo_cards[0].cost
                self._activate_combo(0, reward_accumulator)
                self.combo_cards[0] = None

        # Action 2: Activate second combo card (on press)
        shift_pressed = shift_held and not self.shift_pressed_last_frame
        if shift_pressed and self.combo_cards[1] is not None:
            if self.momentum >= self.combo_cards[1].cost:
                self.momentum -= self.combo_cards[1].cost
                self._activate_combo(1, reward_accumulator)
                self.combo_cards[1] = None
        
        self.space_pressed_last_frame = space_held
        self.shift_pressed_last_frame = shift_held

    def _activate_combo(self, card_index, reward_accumulator):
        reward_accumulator['combo_used'] += 1
        card_name = self.combo_cards[card_index].name
        
        if card_name == "EMP Blast":
            # sfx: emp_blast.wav
            center = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
            self._create_particles(center, 50, self.COLOR_COMBO, 5, life=60, radius=self.WIDTH/2)
            for enemy in self.enemy_units:
                if enemy.pos.distance_to(center) < self.WIDTH / 2:
                    is_destroyed = enemy.take_damage(30, reward_accumulator)
                    if is_destroyed:
                        reward_accumulator['enemy_destroyed'] += 1
                        self._create_particles(enemy.pos, 30, self.COLOR_ENEMY, 2)

        elif card_name == "Overcharge":
            # sfx: overcharge.wav
            for unit in self.player_units:
                unit.is_overcharged = True
                unit.overcharge_timer = self.FPS * 5 # 5 seconds
                self._create_particles(unit.pos, 10, self.COLOR_MOMENTUM, 0.5, life=self.FPS * 5, is_aura=True)
        
        # Regenerate a new card after a delay
        pygame.time.set_timer(pygame.USEREVENT + card_index, 5000, loops=1)


    def _update_game_state(self, reward_accumulator):
        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.current_enemy_spawn_rate = max(0.1, self.current_enemy_spawn_rate - 0.05)
        if self.steps > 0 and self.steps % 500 == 0:
            self.current_enemy_health *= 1.01

        # Enemy spawning
        self.enemy_spawn_timer += 1 / self.FPS
        if self.enemy_spawn_timer >= self.current_enemy_spawn_rate:
            self.enemy_spawn_timer = 0
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top': self._spawn_unit('enemy', random.randint(0, self.WIDTH), 0)
            if edge == 'bottom': self._spawn_unit('enemy', random.randint(0, self.WIDTH), self.HEIGHT)
            if edge == 'left': self._spawn_unit('enemy', 0, random.randint(0, self.HEIGHT))
            if edge == 'right': self._spawn_unit('enemy', self.WIDTH, random.randint(0, self.HEIGHT))
        
        # Update units and handle attacks
        newly_dead_units = set()
        self.projectiles.clear()
        
        all_units = self.player_units + self.enemy_units
        for unit in all_units:
            enemies = self.enemy_units if unit.team == 'player' else self.player_units
            start_pos, end_pos = unit.update(enemies, reward_accumulator)
            if start_pos and unit.target:
                # sfx: laser_shoot.wav
                self.projectiles.append({'start': start_pos, 'end': end_pos, 'team': unit.team, 'timer': 3})
                is_destroyed = unit.target.take_damage(10, reward_accumulator)
                self._create_particles(end_pos, 5, self.COLOR_TEXT, 1)
                if is_destroyed:
                    newly_dead_units.add(unit.target.id)
                    if unit.target.team == 'enemy':
                        reward_accumulator['enemy_destroyed'] += 1
                    # sfx: explosion.wav
                    self._create_particles(unit.target.pos, 30, self.COLOR_ENEMY if unit.target.team == 'enemy' else self.COLOR_PLAYER, 2)

        # Clean up dead units
        self.player_units = [u for u in self.player_units if u.id not in newly_dead_units]
        self.enemy_units = [u for u in self.enemy_units if u.id not in newly_dead_units]

        # Update sectors
        for sector in self.sectors:
            sector.update_owner(self.player_units, self.enemy_units, reward_accumulator)

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] = max(0, p['size'] - p['decay'])

        # Update momentum and combo cards
        self.momentum = max(0, self.momentum - 0.05) # Slow decay
        self.momentum += reward_accumulator.pop('momentum_gain', 0)
        self.momentum = min(self.max_momentum, self.momentum)
        
        # Handle pygame user events for card regeneration
        for event in pygame.event.get():
            if event.type >= pygame.USEREVENT and event.type < pygame.USEREVENT + self.combo_card_slots:
                card_index = event.type - pygame.USEREVENT
                if self.combo_cards[card_index] is None:
                    self._generate_combo_card(card_index)

    def _spawn_unit(self, team, x, y):
        unit_id = self._get_next_unit_id()
        if team == 'player':
            unit = Unit(x, y, 'player', 100, unit_id)
            self.player_units.append(unit)
        else:
            unit = Unit(x, y, 'enemy', self.current_enemy_health, unit_id)
            self.enemy_units.append(unit)

    def _generate_combo_card(self, slot_index):
        if self.combo_cards[slot_index] is None:
            card_type = random.choice(["EMP Blast", "Overcharge"])
            cost = 75 if card_type == "EMP Blast" else 60
            self.combo_cards[slot_index] = ComboCard(card_type, cost)

    def _calculate_reward(self, acc):
        reward = 0
        reward += acc['damage_dealt'] * 0.1
        reward += acc['enemy_destroyed'] * 5.0
        reward += acc['sector_captured'] * 0.5
        reward += acc['combo_used'] * 10.0
        return reward

    def _check_termination(self):
        if not self.player_units:
            return True
        if all(s.owner == 'player' for s in self.sectors):
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "momentum": self.momentum}

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw sectors
        for sector in self.sectors:
            sector.draw(self.screen)
            
        # Draw particles
        for p in self.particles:
            if p.get('is_aura', False):
                self._draw_glowing_circle(self.screen, p['color'], p['pos'], p['size'], 0.5)
            else:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), p['color'])

        # Draw projectiles
        for proj in self.projectiles:
            proj['timer'] -= 1
            if proj['timer'] > 0:
                color = self.COLOR_PLAYER if proj['team'] == 'player' else self.COLOR_ENEMY
                pygame.draw.line(self.screen, color, proj['start'], proj['end'], 2)

        # Draw units and health bars
        for unit in self.player_units + self.enemy_units:
            color = self.COLOR_PLAYER if unit.team == 'player' else self.COLOR_ENEMY
            pos = (int(unit.pos.x), int(unit.pos.y))
            
            # Glow effect
            glow_color = (*color, 70) if not unit.is_overcharged else (*self.COLOR_MOMENTUM, 150)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], unit.radius, glow_color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], unit.radius, glow_color)
            
            # Main body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], unit.radius - 3, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], unit.radius - 3, color)

            # Health bar
            bar_w, bar_h = 24, 4
            bar_x, bar_y = pos[0] - bar_w / 2, pos[1] - unit.radius - 8
            health_pct = max(0, unit.health / unit.max_health)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (bar_x, bar_y, bar_w * health_pct, bar_h))

    def _render_ui(self):
        # Score and Steps
        score_text = self.FONT_UI.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        steps_text = self.FONT_UI.render(f"STEPS: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        # Momentum Bar
        bar_w, bar_h = 200, 20
        bar_x, bar_y = (self.WIDTH - bar_w) / 2, 10
        momentum_pct = self.momentum / self.max_momentum
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_MOMENTUM, (bar_x, bar_y, bar_w * momentum_pct, bar_h), border_radius=4)
        
        # Combo Cards
        card_w, card_h = 120, 50
        spacing = 10
        start_x = (self.WIDTH - (self.combo_card_slots * card_w + (self.combo_card_slots - 1) * spacing)) / 2
        
        for i, card in enumerate(self.combo_cards):
            card_x = start_x + i * (card_w + spacing)
            card_y = self.HEIGHT - card_h - 10
            rect = pygame.Rect(card_x, card_y, card_w, card_h)
            
            if card:
                can_afford = self.momentum >= card.cost
                border_color = self.COLOR_COMBO if can_afford else self.COLOR_GRID
                pygame.draw.rect(self.screen, (30, 20, 50, 200), rect, border_radius=5)
                pygame.draw.rect(self.screen, border_color, rect, 2, border_radius=5)
                
                name_text = self.FONT_CARD.render(card.name, True, self.COLOR_TEXT)
                cost_text = self.FONT_CARD.render(f"Cost: {card.cost}", True, self.COLOR_MOMENTUM)
                self.screen.blit(name_text, (rect.centerx - name_text.get_width()/2, rect.y + 8))
                self.screen.blit(cost_text, (rect.centerx - cost_text.get_width()/2, rect.y + 26))
            else:
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 2, border_radius=5)

    def _create_particles(self, pos, count, color, speed_mult, life=20, radius=1, is_aura=False):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 1.5) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            start_pos = pos + vel * random.uniform(0, radius) if radius > 1 else pos
            self.particles.append({
                'pos': pygame.Vector2(start_pos),
                'vel': vel,
                'life': random.randint(life // 2, life),
                'color': color,
                'size': random.uniform(2, 5),
                'decay': 0.1,
                'is_aura': is_aura
            })
            
    def _draw_glowing_circle(self, surface, color, center, radius, glow_strength):
        if radius <= 0: return
        r, g, b = color
        for i in range(int(radius * glow_strength), 0, -1):
            alpha = int(255 * (1 - i / (radius * glow_strength))**2 * 0.3)
            pygame.gfxdraw.aacircle(surface, int(center.x), int(center.y), int(radius + i), (r, g, b, alpha))

    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # This block is for manual testing and visualization.
    # It will not be executed in the evaluation environment.
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'mac'
    pygame.display.init()
    pygame.font.init()

    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Cyberpunk Battlefields")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    movement_map = {
        pygame.K_w: 1, pygame.K_UP: 1,
        pygame.K_s: 2, pygame.K_DOWN: 2,
        pygame.K_a: 3, pygame.K_LEFT: 3,
        pygame.K_d: 4, pygame.K_RIGHT: 4
    }

    running = True
    while running:
        movement = 0
        space_held = 0
        shift_held = 0

        # Poll events before getting key states
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        for key, move_val in movement_map.items():
            if keys[key]:
                movement = move_val
                break # Prioritize first found
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Blit the observation from the environment to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
            obs, info = env.reset()

        clock.tick(env.FPS)
        
    pygame.quit()