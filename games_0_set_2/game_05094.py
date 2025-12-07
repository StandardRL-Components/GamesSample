# Generated: 2025-08-28T03:56:31.598904
# Source Brief: brief_05094.md
# Brief Index: 5094

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Space for Melee, Shift for Ranged, Space+Shift for AOE."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defeat waves of monsters in a grid-based arena. Use tactical attacks to survive."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.MAX_STEPS = 1000
        
        # Calculate grid properties
        self.ARENA_HEIGHT = self.HEIGHT
        self.CELL_SIZE = self.ARENA_HEIGHT // self.GRID_SIZE
        self.ARENA_WIDTH = self.CELL_SIZE * self.GRID_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.ARENA_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.ARENA_HEIGHT) // 2

        # Colors
        self.COLOR_BG = (18, 22, 33)
        self.COLOR_GRID = (40, 48, 61)
        self.COLOR_PLAYER = (57, 255, 20)
        self.COLOR_PLAYER_DMG = (255, 100, 100)
        self.COLOR_MONSTER_CHASER = (255, 56, 56)
        self.COLOR_MONSTER_SHOOTER = (255, 150, 56)
        self.COLOR_PROJECTILE_PLAYER = (137, 207, 240)
        self.COLOR_PROJECTILE_MONSTER = (255, 165, 0)
        self.COLOR_UI_BG = (25, 30, 42)
        self.COLOR_UI_HEALTH = (57, 255, 20)
        self.COLOR_UI_HEALTH_BG = (80, 0, 0)
        self.COLOR_WHITE = (240, 240, 240)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_health = 100
        self.player_max_health = 100
        self.player_facing_dir = (0, -1) # Up
        self.player_took_damage_timer = 0
        self.wave = 1
        self.monsters = []
        self.projectiles = []
        self.particles = []
        
        # This will call reset, so all variables are initialized before validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 1
        
        self.player_pos = [self.GRID_SIZE // 2, self.GRID_SIZE - 1]
        self.player_health = self.player_max_health
        self.player_facing_dir = (0, -1)
        self.player_took_damage_timer = 0
        
        self.monsters = []
        self.projectiles = []
        self.particles = []
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty for time
        
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Player Turn ---
        # 1. Handle Movement
        moved = False
        if movement > 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_x = max(0, min(self.GRID_SIZE - 1, self.player_pos[0] + dx))
            new_y = max(0, min(self.GRID_SIZE - 1, self.player_pos[1] + dy))
            if [new_x, new_y] != self.player_pos:
                self.player_pos = [new_x, new_y]
                moved = True
            if (dx, dy) != (0, 0):
                self.player_facing_dir = (dx, dy)

        # 2. Handle Attacks
        attack_type = "none"
        if space_held and shift_held: attack_type = "aoe"
        elif space_held: attack_type = "melee"
        elif shift_held: attack_type = "ranged"

        if attack_type != "none":
            reward += self._player_attack(attack_type)

        # --- Game Logic Update ---
        # 1. Update Projectiles
        projectile_hits = self._update_projectiles()
        for hit in projectile_hits:
            if hit['target'] == 'player':
                self.player_health -= hit['damage']
                self.player_took_damage_timer = 10 # frames
                self._create_particles(self.player_pos, self.COLOR_PLAYER_DMG, 20, is_grid_pos=True)
                # sfx: player_hurt
            elif hit['target'] == 'monster':
                hit['monster']['health'] -= hit['damage']
                reward += 0.2 # Reward for hitting
                self._create_particles(hit['monster']['pos'], self.COLOR_PROJECTILE_PLAYER, 15, is_grid_pos=True)
                # sfx: monster_hit
        
        # --- Monster Turn ---
        if not moved and attack_type == "none": # Monsters only act if player waits
            pass # No monster turn for now to keep it simpler, they only react by shooting
        else:
             for monster in self.monsters:
                self._monster_action(monster)

        # --- Cleanup and State Update ---
        # 1. Remove dead monsters
        dead_monsters = [m for m in self.monsters if m['health'] <= 0]
        for monster in dead_monsters:
            self.score += 10
            reward += 10 # Reward for defeating
            self._create_particles(monster['pos'], monster['color'], 40, is_grid_pos=True, life=25)
            # sfx: monster_death_explosion
        self.monsters = [m for m in self.monsters if m['health'] > 0]

        # 2. Check for wave clear
        if not self.monsters:
            self.wave += 1
            reward += 100 # Big reward for clearing wave
            self._spawn_wave()
            # sfx: wave_clear_success

        # 3. Update timers and particles
        self.player_took_damage_timer = max(0, self.player_took_damage_timer - 1)
        self._update_particles()
        
        # --- Termination Check ---
        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.player_health <= 0:
                reward -= 50 # Penalty for dying
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _player_attack(self, attack_type):
        reward = 0
        if attack_type == "melee":
            # sfx: melee_swing
            target_pos = [self.player_pos[0] + self.player_facing_dir[0], self.player_pos[1] + self.player_facing_dir[1]]
            self._create_particles(target_pos, self.COLOR_WHITE, 10, is_grid_pos=True, life=8, speed=2)
            for m in self.monsters:
                if m['pos'] == target_pos:
                    m['health'] -= 10
                    reward += 0.2
                    # sfx: melee_hit
        elif attack_type == "ranged":
            # sfx: ranged_fire
            self.projectiles.append({
                'pos': list(self.player_pos),
                'dir': self.player_facing_dir,
                'speed': 0.5, 'owner': 'player', 'damage': 8
            })
        elif attack_type == "aoe":
            # sfx: aoe_charge_and_blast
            aoe_center_px = self._grid_to_pixel(self.player_pos)
            self.particles.append({'pos': aoe_center_px, 'rad': 0, 'max_rad': self.CELL_SIZE*1.5, 'life': 10, 'max_life': 10, 'color': self.COLOR_PROJECTILE_PLAYER, 'type': 'shockwave'})
            for m in self.monsters:
                dist = max(abs(m['pos'][0] - self.player_pos[0]), abs(m['pos'][1] - self.player_pos[1]))
                if dist <= 1:
                    m['health'] -= 15
                    reward += 0.2
        return reward

    def _monster_action(self, monster):
        player_dist = max(abs(monster['pos'][0] - self.player_pos[0]), abs(monster['pos'][1] - self.player_pos[1]))
        
        if monster['type'] == 'chaser':
            if player_dist == 1: # Melee attack if adjacent
                self.player_health -= 5
                self.player_took_damage_timer = 10
                self._create_particles(self.player_pos, monster['color'], 15, is_grid_pos=True)
                # sfx: monster_melee_hit
            elif player_dist > 1: # Move towards player
                dx = np.sign(self.player_pos[0] - monster['pos'][0])
                dy = np.sign(self.player_pos[1] - monster['pos'][1])
                # Prioritize one axis to avoid diagonal movement
                if dx != 0 and dy != 0:
                    if self.np_random.random() > 0.5: dx = 0
                    else: dy = 0
                
                new_pos = [monster['pos'][0] + dx, monster['pos'][1] + dy]
                if self._is_pos_empty(new_pos):
                    monster['pos'] = new_pos

        elif monster['type'] == 'shooter':
            if player_dist > 2 and (monster['pos'][0] == self.player_pos[0] or monster['pos'][1] == self.player_pos[1]):
                # Fire if player is in line of sight and not too close
                if self.np_random.random() < 0.5: # 50% chance to fire
                    # sfx: monster_ranged_fire
                    direction = (np.sign(self.player_pos[0] - monster['pos'][0]), np.sign(self.player_pos[1] - monster['pos'][1]))
                    self.projectiles.append({
                        'pos': list(monster['pos']),
                        'dir': direction, 'speed': 0.3, 'owner': 'monster', 'damage': 10
                    })
            else: # Move to a better position or randomly
                if self.np_random.random() < 0.3:
                    moves = [(0,1), (0,-1), (1,0), (-1,0)]
                    move = random.choice(moves)
                    new_pos = [monster['pos'][0] + move[0], monster['pos'][1] + move[1]]
                    if self._is_pos_valid(new_pos) and self._is_pos_empty(new_pos):
                        monster['pos'] = new_pos

    def _is_pos_valid(self, pos):
        return 0 <= pos[0] < self.GRID_SIZE and 0 <= pos[1] < self.GRID_SIZE

    def _is_pos_empty(self, pos):
        if not self._is_pos_valid(pos): return False
        if pos == self.player_pos: return False
        for m in self.monsters:
            if m['pos'] == pos: return False
        return True

    def _spawn_wave(self):
        monster_health = 20 + (self.wave - 1) * 2
        for _ in range(5):
            spawn_pos = [self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE // 2)]
            while not self._is_pos_empty(spawn_pos):
                spawn_pos = [self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE // 2)]
            
            monster_type = self.np_random.choice(['chaser', 'shooter'])
            color = self.COLOR_MONSTER_CHASER if monster_type == 'chaser' else self.COLOR_MONSTER_SHOOTER
            self.monsters.append({'pos': spawn_pos, 'health': monster_health, 'max_health': monster_health, 'type': monster_type, 'color': color})

    def _update_projectiles(self):
        hits = []
        for p in self.projectiles[:]:
            p['pos'][0] += p['dir'][0] * p['speed']
            p['pos'][1] += p['dir'][1] * p['speed']
            
            grid_pos = [round(p['pos'][0]), round(p['pos'][1])]
            
            if not self._is_pos_valid(grid_pos):
                self.projectiles.remove(p)
                continue
            
            if p['owner'] == 'player':
                for m in self.monsters:
                    if m['pos'] == grid_pos:
                        hits.append({'target': 'monster', 'monster': m, 'damage': p['damage']})
                        self.projectiles.remove(p)
                        break
            elif p['owner'] == 'monster':
                if self.player_pos == grid_pos:
                    hits.append({'target': 'player', 'damage': p['damage']})
                    self.projectiles.remove(p)
        return hits

    def _create_particles(self, pos, color, count, is_grid_pos=False, life=15, speed=3):
        start_pos = self._grid_to_pixel(pos) if is_grid_pos else pos
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            vel = self.np_random.random() * speed
            self.particles.append({
                'pos': list(start_pos),
                'vel': [math.cos(angle) * vel, math.sin(angle) * vel],
                'life': life, 'max_life': life, 'color': color, 'type': 'particle'
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
                continue
            
            if p['type'] == 'particle':
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
            elif p['type'] == 'shockwave':
                p['rad'] += p['max_rad'] / p['life'] if p['life'] > 0 else 0


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave, "health": self.player_health}
    
    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return [x, y]

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            y = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.ARENA_HEIGHT))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.ARENA_WIDTH, y))
        
        # Draw monsters
        for m in self.monsters:
            pos_px = self._grid_to_pixel(m['pos'])
            size = int(self.CELL_SIZE * 0.7)
            rect = pygame.Rect(pos_px[0] - size//2, pos_px[1] - size//2, size, size)
            pygame.draw.rect(self.screen, m['color'], rect, border_radius=4)
            # Health bar for monster
            hp_ratio = m['health'] / m['max_health']
            bar_w = int(size * 0.8)
            bar_h = 4
            pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_BG, (pos_px[0]-bar_w//2, pos_px[1]-size//2-8, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, (pos_px[0]-bar_w//2, pos_px[1]-size//2-8, int(bar_w*hp_ratio), bar_h))

        # Draw player
        player_pos_px = self._grid_to_pixel(self.player_pos)
        p_color = self.COLOR_PLAYER_DMG if self.player_took_damage_timer > 0 else self.COLOR_PLAYER
        size = int(self.CELL_SIZE * 0.8)
        player_rect = pygame.Rect(player_pos_px[0] - size//2, player_pos_px[1] - size//2, size, size)
        pygame.draw.rect(self.screen, p_color, player_rect, border_radius=6)
        # Draw facing indicator (eye)
        eye_pos = (player_pos_px[0] + self.player_facing_dir[0] * size * 0.25, 
                   player_pos_px[1] + self.player_facing_dir[1] * size * 0.25)
        pygame.draw.circle(self.screen, self.COLOR_BG, eye_pos, size * 0.1)

        # Draw projectiles
        for p in self.projectiles:
            pos_px = self._grid_to_pixel(p['pos'])
            color = self.COLOR_PROJECTILE_PLAYER if p['owner'] == 'player' else self.COLOR_PROJECTILE_MONSTER
            pygame.draw.circle(self.screen, color, pos_px, self.CELL_SIZE * 0.15)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color_with_alpha = p['color'] + (alpha,)
            if p['type'] == 'particle':
                size = max(1, int(8 * (p['life'] / p['max_life'])))
                # This drawing method might not support alpha on all systems, but we follow the original intent.
                # A temporary surface would be a more robust solution.
                temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
                temp_surf.fill(color_with_alpha)
                self.screen.blit(temp_surf, (p['pos'][0]-size//2, p['pos'][1]-size//2), special_flags=pygame.BLEND_RGBA_ADD)
            elif p['type'] == 'shockwave':
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['rad']), color_with_alpha)


    def _render_ui(self):
        # Health Bar
        bar_x, bar_y, bar_w, bar_h = 10, 10, 200, 25
        hp_ratio = max(0, self.player_health / self.player_max_health)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_BG, (bar_x+2, bar_y+2, bar_w-4, bar_h-4))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, (bar_x+2, bar_y+2, (bar_w-4) * hp_ratio, bar_h-4))
        health_text = self.font_small.render(f"HP: {int(self.player_health)}/{self.player_max_health}", True, self.COLOR_WHITE)
        self.screen.blit(health_text, (bar_x + 5, bar_y + 5))

        # Score and Wave
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        wave_text = self.font_main.render(f"WAVE: {self.wave}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 35))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            end_text = self.font_main.render("GAME OVER", True, self.COLOR_MONSTER_CHASER)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # This will reset the env, and needs to be done before validation
        obs, info = self.reset()

        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: Gymnasium's 'human' render mode is not used here for direct control.
    # We are calling the environment's methods and rendering manually.
    
    # Re-enable video driver for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup pygame window for human play
    pygame.display.set_caption("Grid Monster Slayer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print("\n" + "="*30)
    print(f"GAME: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("="*30 + "\n")
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
            action = [movement, space_held, shift_held]
            
            # Since auto_advance is False, we only step when there's an action or keypress
            # For human play, we step every frame to feel responsive.
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != -0.01: # Don't print the time penalty
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Control the speed of human play (10 actions per second)

    env.close()