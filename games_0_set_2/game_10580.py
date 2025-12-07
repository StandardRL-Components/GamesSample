import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()
pygame.font.init()

from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Purify a corrupted digital system by clearing sectors of viruses and bugs. "
        "Use your sword and a gravity pulse ability to survive and cleanse the grid."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to swing your sword and "
        "shift to activate the gravity pulse ability."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Sizing
    WIDTH, HEIGHT = 640, 400
    UI_HEIGHT = 50
    GAME_HEIGHT = HEIGHT - UI_HEIGHT

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 75, 128)
    COLOR_ENEMY_VIRUS = (255, 50, 50)
    COLOR_ENEMY_BUG = (200, 0, 0)
    COLOR_ENEMY_GLOW = (100, 0, 0)
    COLOR_FRAGMENT = (255, 215, 0)
    COLOR_FRAGMENT_GLOW = (128, 108, 0)
    COLOR_SECTOR_CORRUPTED = (100, 0, 20, 100)
    COLOR_SWORD = (200, 220, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BG = (25, 20, 40)
    COLOR_HEALTH_BAR = (50, 200, 50)
    COLOR_HEALTH_BAR_BG = (100, 50, 50)

    # Game Parameters
    MAX_STEPS = 2000
    PLAYER_SPEED = 4.0
    PLAYER_DRAG = 0.85
    PLAYER_HEALTH = 100
    PLAYER_RADIUS = 12

    ENEMY_VIRUS_HEALTH = 20
    ENEMY_VIRUS_SPEED = 1.0
    ENEMY_VIRUS_RADIUS = 8
    ENEMY_BUG_HEALTH = 60
    ENEMY_BUG_SPEED = 0.5
    ENEMY_BUG_RADIUS = 15

    SWORD_COOLDOWN = 15  # steps
    SWORD_DURATION = 5   # steps
    SWORD_RANGE = 40
    SWORD_ANGLE = 90
    SWORD_DAMAGE = 10

    ABILITY_COOLDOWN = 90 # steps
    ABILITY_GRAVITY_PULSE_RADIUS = 150
    ABILITY_GRAVITY_PULSE_STRENGTH = 15

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- GYM SPACES ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- PYGAME SETUP ---
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_main = pygame.font.SysFont("Consolas", 20)
            self.font_title = pygame.font.SysFont("Consolas", 28, bold=True)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 24)
            self.font_title = pygame.font.SysFont(None, 32, bold=True)

        # --- STATE VARIABLES ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0

        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_health = 0
        self.last_move_dir = pygame.math.Vector2(1, 0)

        self.enemies = []
        self.sectors = []
        self.fragments = []
        self.particles = []
        
        self.sword_state = {'active': False, 'timer': 0, 'angle': 0}
        self.sword_cooldown_timer = 0
        self.ability_cooldown_timer = 0

        self.prev_space_held = False
        self.prev_shift_held = False


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.GAME_HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_health = self.PLAYER_HEALTH
        self.last_move_dir = pygame.math.Vector2(1, 0)

        self.enemies = []
        self.particles = []
        self.sword_cooldown_timer = 0
        self.ability_cooldown_timer = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward_this_step = 0

        self._handle_input(action)
        self._update_player()
        self._update_sword()
        self._update_enemies()
        self._update_particles()
        self._update_sectors()
        self._check_collisions()
        self._spawn_enemies()
        
        self._update_cooldowns()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        # Final reward calculation
        reward = self.reward_this_step
        if terminated and not truncated:
            if self.player_health <= 0:
                reward -= 100
            elif self._all_sectors_purified():
                reward += 100
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- UPDATE LOGIC ---

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1  # Right
        
        if move_vec.length_squared() > 0:
            move_vec.normalize_ip()
            self.player_vel += move_vec * self.PLAYER_SPEED
            self.last_move_dir = move_vec.copy()

        # Rising edge detection for actions
        if space_held and not self.prev_space_held and self.sword_cooldown_timer <= 0:
            self._activate_sword() # SFX: Sword_Slash.wav
        if shift_held and not self.prev_shift_held and self.ability_cooldown_timer <= 0:
            self._activate_ability() # SFX: Ability_Activate.wav
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_player(self):
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_DRAG

        # Boundary checks
        self.player_pos.x = max(self.PLAYER_RADIUS, min(self.WIDTH - self.PLAYER_RADIUS, self.player_pos.x))
        self.player_pos.y = max(self.PLAYER_RADIUS, min(self.GAME_HEIGHT - self.PLAYER_RADIUS, self.player_pos.y))
    
    def _update_cooldowns(self):
        if self.sword_cooldown_timer > 0:
            self.sword_cooldown_timer -= 1
        if self.ability_cooldown_timer > 0:
            self.ability_cooldown_timer -= 1

    def _activate_sword(self):
        self.sword_state['active'] = True
        self.sword_state['timer'] = self.SWORD_DURATION
        self.sword_state['angle'] = self.last_move_dir.angle_to(pygame.math.Vector2(1, 0))
        self.sword_cooldown_timer = self.SWORD_COOLDOWN

    def _update_sword(self):
        if self.sword_state['active']:
            self.sword_state['timer'] -= 1
            if self.sword_state['timer'] <= 0:
                self.sword_state['active'] = False

    def _activate_ability(self):
        self.ability_cooldown_timer = self.ABILITY_COOLDOWN
        # SFX: Gravity_Pulse.wav
        self._create_shockwave_particles(self.player_pos, self.ABILITY_GRAVITY_PULSE_RADIUS, self.COLOR_PLAYER)
        for enemy in self.enemies:
            vec_to_enemy = enemy['pos'] - self.player_pos
            dist = vec_to_enemy.length()
            if 0 < dist < self.ABILITY_GRAVITY_PULSE_RADIUS:
                repel_strength = self.ABILITY_GRAVITY_PULSE_STRENGTH * (1 - (dist / self.ABILITY_GRAVITY_PULSE_RADIUS))
                enemy['vel'] += vec_to_enemy.normalize() * repel_strength

    def _update_enemies(self):
        # Difficulty scaling
        speed_bonus = 0.05 * (self.steps // 200)
        
        for enemy in self.enemies:
            # Movement AI
            direction = (self.player_pos - enemy['pos']).normalize()
            enemy['vel'] += direction * (enemy['speed'] + speed_bonus)
            
            # Drag
            enemy['pos'] += enemy['vel']
            enemy['vel'] *= 0.8

            # Boundary checks
            enemy['pos'].x = max(enemy['radius'], min(self.WIDTH - enemy['radius'], enemy['pos'].x))
            enemy['pos'].y = max(enemy['radius'], min(self.GAME_HEIGHT - enemy['radius'], enemy['pos'].y))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _update_sectors(self):
        for sector in self.sectors:
            sector['enemies'] = [e for e in self.enemies if pygame.Rect(sector['rect']).collidepoint(e['pos'])]
            
            if not sector['purified'] and not sector['enemies'] and sector['fully_spawned']:
                sector['purified'] = True
                self.reward_this_step += 5
                # SFX: Sector_Purified.wav
                self._create_explosion(pygame.Rect(sector['rect']).center, self.COLOR_FRAGMENT, 50)


    def _check_collisions(self):
        # Player <-> Enemy
        for enemy in self.enemies:
            if self.player_pos.distance_to(enemy['pos']) < self.PLAYER_RADIUS + enemy['radius']:
                self.player_health = max(0, self.player_health - 1) # Small continuous damage
                self.reward_this_step -= 0.1
                if self.player_health > 0:
                    self._create_hit_particles(self.player_pos, self.COLOR_ENEMY_VIRUS)
                # SFX: Player_Hurt.wav
        
        # Sword <-> Enemy
        if self.sword_state['active']:
            for enemy in self.enemies:
                if 'hit_by_sword' not in enemy or not enemy['hit_by_sword']:
                    vec_to_enemy = enemy['pos'] - self.player_pos
                    dist = vec_to_enemy.length()
                    if 0 < dist < self.PLAYER_RADIUS + self.SWORD_RANGE:
                        angle_to_enemy = self.last_move_dir.angle_to(vec_to_enemy)
                        if abs(angle_to_enemy) < self.SWORD_ANGLE / 2:
                            enemy['health'] -= self.SWORD_DAMAGE
                            self.reward_this_step += 0.1
                            enemy['hit_by_sword'] = True
                            self._create_hit_particles(enemy['pos'], self.COLOR_SWORD)
                            # SFX: Enemy_Hit.wav
        else: # Reset hit flag after swing
            for enemy in self.enemies:
                enemy['hit_by_sword'] = False

        # Check for dead enemies
        dead_enemies = [e for e in self.enemies if e['health'] <= 0]
        for dead_enemy in dead_enemies:
            self._create_explosion(dead_enemy['pos'], dead_enemy['color'], 20)
            # SFX: Enemy_Die.wav
        self.enemies = [e for e in self.enemies if e['health'] > 0]

        # Player <-> Fragments
        fragments_to_remove = []
        for frag in self.fragments:
            if self.player_pos.distance_to(frag['pos']) < self.PLAYER_RADIUS + frag['radius']:
                fragments_to_remove.append(frag)
                self.reward_this_step += 1
                self.score += 10 # A bonus score for collection
                self._create_explosion(frag['pos'], self.COLOR_FRAGMENT, 30)
                # SFX: Fragment_Collect.wav
        self.fragments = [f for f in self.fragments if f not in fragments_to_remove]
    
    def _spawn_enemies(self):
        spawn_rate_mod = 0.01 * (self.steps // 100)
        for sector in self.sectors:
            if not sector['purified'] and not sector['fully_spawned']:
                if self.np_random.random() < sector['spawn_rate'] + spawn_rate_mod:
                    if sector['spawns_left'] > 0:
                        self._create_enemy(sector)
                        sector['spawns_left'] -= 1
                        if sector['spawns_left'] == 0:
                            sector['fully_spawned'] = True
    
    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            return True
        if self._all_sectors_purified():
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _all_sectors_purified(self):
        return all(s['purified'] for s in self.sectors)

    # --- RENDERING ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_effects()
        self._render_sectors()
        self._render_fragments()
        self._render_particles()
        self._render_enemies()
        self._render_player()
        self._render_sword()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_effects(self):
        # Simple starfield
        for i in range(50):
            x = (i * 37 + self.steps // 10) % self.WIDTH
            y = (i * 141) % self.GAME_HEIGHT
            size = (i % 3)
            pygame.draw.rect(self.screen, (50, 50, 70), (x, y, size, size))

    def _render_sectors(self):
        for sector in self.sectors:
            if not sector['purified']:
                s = pygame.Surface(sector['rect'].size, pygame.SRCALPHA)
                s.fill(self.COLOR_SECTOR_CORRUPTED)
                self.screen.blit(s, sector['rect'].topleft)

    def _render_fragments(self):
        for frag in self.fragments:
            self._render_glow_circle(frag['pos'], frag['radius'] * 2, self.COLOR_FRAGMENT_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, int(frag['pos'].x), int(frag['pos'].y), frag['radius'], self.COLOR_FRAGMENT)

    def _render_enemies(self):
        for enemy in self.enemies:
            self._render_glow_circle(enemy['pos'], enemy['radius'] * 1.5, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, int(enemy['pos'].x), int(enemy['pos'].y), enemy['radius'], enemy['color'])
            pygame.gfxdraw.aacircle(self.screen, int(enemy['pos'].x), int(enemy['pos'].y), enemy['radius'], enemy['color'])

    def _render_player(self):
        pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        bob = int(math.sin(self.steps * 0.2) * 2)
        
        self._render_glow_circle(self.player_pos, self.PLAYER_RADIUS * 2.5, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1] + bob, self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1] + bob, self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_sword(self):
        if self.sword_state['active']:
            angle = self.sword_state['angle']
            progress = 1.0 - (self.sword_state['timer'] / self.SWORD_DURATION)
            
            start_angle = math.radians(-angle - self.SWORD_ANGLE / 2)
            end_angle = math.radians(-angle + self.SWORD_ANGLE / 2)
            
            # Interpolate angle for swoosh effect
            current_end_angle = start_angle + (end_angle - start_angle) * progress

            rect = pygame.Rect(
                self.player_pos.x - self.SWORD_RANGE, self.player_pos.y - self.SWORD_RANGE,
                self.SWORD_RANGE * 2, self.SWORD_RANGE * 2
            )
            pygame.draw.arc(self.screen, self.COLOR_SWORD, rect, start_angle, current_end_angle, 4)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (p['pos'].x - p['size'], p['pos'].y - p['size']))

    def _render_ui(self):
        # UI Background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, self.GAME_HEIGHT, self.WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_PLAYER_GLOW, (0, self.GAME_HEIGHT), (self.WIDTH, self.GAME_HEIGHT), 2)
        
        # Health Bar
        health_pct = max(0, self.player_health / self.PLAYER_HEALTH)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, self.GAME_HEIGHT + 15, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, self.GAME_HEIGHT + 15, int(bar_width * health_pct), 20))
        health_text = self.font_main.render(f"HP", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, self.GAME_HEIGHT + 16))

        # Score
        score_text = self.font_title.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, self.GAME_HEIGHT + 10))

        # Ability Cooldown
        ability_text = self.font_main.render("PULSE", True, self.COLOR_UI_TEXT)
        self.screen.blit(ability_text, (self.WIDTH - 150, self.GAME_HEIGHT + 16))
        
        cooldown_pct = 1.0 - max(0, self.ability_cooldown_timer / self.ABILITY_COOLDOWN)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GLOW, (self.WIDTH - 90, self.GAME_HEIGHT + 15, 80, 20))
        if cooldown_pct > 0:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (self.WIDTH - 90, self.GAME_HEIGHT + 15, int(80 * cooldown_pct), 20))


    def _render_glow_circle(self, pos, radius, color):
        s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, color, (radius, radius), radius)
        self.screen.blit(s, (int(pos.x - radius), int(pos.y - radius)), special_flags=pygame.BLEND_RGBA_ADD)

    # --- HELPERS ---

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _generate_level(self):
        self.sectors = [
            {'rect': pygame.Rect(50, 50, 200, 120), 'spawn_rate': 0.02, 'spawns_left': 5, 'max_spawns': 5, 'purified': False, 'fully_spawned': False, 'enemies': []},
            {'rect': pygame.Rect(390, 50, 200, 120), 'spawn_rate': 0.02, 'spawns_left': 5, 'max_spawns': 5, 'purified': False, 'fully_spawned': False, 'enemies': []},
            {'rect': pygame.Rect(220, 200, 200, 120), 'spawn_rate': 0.03, 'spawns_left': 8, 'max_spawns': 8, 'purified': False, 'fully_spawned': False, 'enemies': []},
        ]
        self.fragments = [
            {'pos': pygame.math.Vector2(self.np_random.uniform(100, 540), self.np_random.uniform(50, 300)), 'radius': 8}
        ]
        # Ensure fragment doesn't spawn on player
        while self.fragments[0]['pos'].distance_to(self.player_pos) < 100:
            self.fragments[0]['pos'] = pygame.math.Vector2(self.np_random.uniform(100, 540), self.np_random.uniform(50, 300))
        
        # Initial spawn
        self._create_enemy(self.sectors[0])
        self.sectors[0]['spawns_left'] -= 1

    def _create_enemy(self, sector):
        spawn_pos = pygame.math.Vector2(
            self.np_random.uniform(sector['rect'].left, sector['rect'].right),
            self.np_random.uniform(sector['rect'].top, sector['rect'].bottom)
        )
        
        enemy_type = 'virus' if self.np_random.random() < 0.7 else 'bug'
        
        if enemy_type == 'virus':
            self.enemies.append({
                'pos': spawn_pos, 'vel': pygame.math.Vector2(0,0), 'type': 'virus',
                'health': self.ENEMY_VIRUS_HEALTH, 'speed': self.ENEMY_VIRUS_SPEED,
                'radius': self.ENEMY_VIRUS_RADIUS, 'color': self.COLOR_ENEMY_VIRUS
            })
        else:
            self.enemies.append({
                'pos': spawn_pos, 'vel': pygame.math.Vector2(0,0), 'type': 'bug',
                'health': self.ENEMY_BUG_HEALTH, 'speed': self.ENEMY_BUG_SPEED,
                'radius': self.ENEMY_BUG_RADIUS, 'color': self.COLOR_ENEMY_BUG
            })

    def _create_hit_particles(self, pos, color):
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': self.np_random.integers(10, 20),
                'max_life': 20,
                'color': color,
                'size': self.np_random.integers(2, 4)
            })

    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': self.np_random.integers(20, 40),
                'max_life': 40,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })
    
    def _create_shockwave_particles(self, pos, radius, color):
        for i in range(36):
            angle = math.radians(i * 10)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.math.Vector2(math.cos(angle), math.sin(angle)) * (radius / 15),
                'life': 15,
                'max_life': 15,
                'color': color,
                'size': 3
            })

if __name__ == "__main__":
    # --- Manual Play ---
    # The main loop needs a visible display, so we unset the dummy driver
    os.environ["SDL_VIDEODRIVER"] = ""
    pygame.quit() # Quit the dummy instance
    pygame.init() # Re-init with a real display driver
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("GameEnv")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Run at 30 FPS
        
    pygame.quit()