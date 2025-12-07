
# Generated: 2025-08-28T02:53:32.970056
# Source Brief: brief_04600.md
# Brief Index: 4600

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Space to fire a ranged shot in your facing direction. "
        "Shift to unleash a close-range shockwave. Moving or attacking uses your turn."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a combat mech in a grid-based arena. Battle against 15 hostile robots in "
        "turn-based combat. Use strategic positioning and a mix of ranged and melee attacks to survive and dominate."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_W = 16
        self.GRID_H = 10
        self.CELL_SIZE = 40
        self.MAX_STEPS = 1000
        self.NUM_ENEMIES = 15
        self.PLAYER_MAX_HEALTH = 3

        # --- Colors ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_PROJECTILE = (100, 200, 255)
        self.COLOR_EXPLOSION_1 = (255, 200, 50)
        self.COLOR_EXPLOSION_2 = (255, 100, 0)
        self.COLOR_SHOCKWAVE = (200, 150, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_BAR_BG = (80, 20, 20)
        self.COLOR_HEALTH_BAR = (50, 220, 50)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # Initialize state variables
        self.reset()

        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_facing = 1  # 1:Up, 2:Down, 3:Left, 4:Right
        self.player_damage_flash = 0

        self.enemies = []
        self._spawn_enemies(self.NUM_ENEMIES)

        self.projectiles = []
        self.effects = []  # For explosions, shockwaves, etc.
        
        return self._get_observation(), self._get_info()

    def _spawn_enemies(self, num_enemies):
        occupied_cells = {tuple(self.player_pos)}
        for _ in range(num_enemies):
            pos = (self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H))
            while pos in occupied_cells:
                pos = (self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H))
            occupied_cells.add(pos)
            self.enemies.append({'pos': list(pos), 'health': 1, 'damage_flash': 0})
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.1  # Cost of existing per turn
        
        # --- Player Turn ---
        enemies_defeated_this_turn = self._handle_player_action(action)
        reward += enemies_defeated_this_turn * 10
        self.score += enemies_defeated_this_turn

        # --- World Turn ---
        if not self.game_over:
            self._update_projectiles()
            self._update_enemies()

        # --- Update Visual Effects ---
        self._update_effects()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.win:
                reward += 100  # Victory bonus
            else:
                reward -= 100 # Defeat penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        enemies_defeated = 0

        # Priority: Movement > Melee > Ranged > Wait
        if movement != 0:
            # --- Handle Movement ---
            self.player_facing = movement
            new_pos = list(self.player_pos)
            if movement == 1: new_pos[1] -= 1 # Up
            elif movement == 2: new_pos[1] += 1 # Down
            elif movement == 3: new_pos[0] -= 1 # Left
            elif movement == 4: new_pos[0] += 1 # Right
            
            # Check boundaries and collisions
            if 0 <= new_pos[0] < self.GRID_W and 0 <= new_pos[1] < self.GRID_H:
                if not any(e['pos'] == new_pos for e in self.enemies):
                    self.player_pos = new_pos
        
        elif shift_held:
            # --- Handle Melee Attack ---
            # sfx: shockwave_charge.wav, shockwave_release.wav
            px, py = self.player_pos
            self.effects.append({'type': 'shockwave', 'pos': [px, py], 'radius': 0, 'max_radius': 1.5, 'timer': 10})
            
            enemies_to_remove = []
            for enemy in self.enemies:
                ex, ey = enemy['pos']
                if abs(px - ex) <= 1 and abs(py - ey) <= 1 and (px, py) != (ex, ey):
                    enemy['health'] -= 1
                    if enemy['health'] <= 0:
                        enemies_to_remove.append(enemy)
                        enemies_defeated += 1
                        # sfx: explosion.wav
                        self.effects.append({'type': 'explosion', 'pos': enemy['pos'], 'timer': 20, 'particles': []})
            self.enemies = [e for e in self.enemies if e not in enemies_to_remove]

        elif space_held:
            # --- Handle Ranged Attack ---
            # sfx: laser_shoot.wav
            px, py = self.player_pos
            self.effects.append({'type': 'muzzle_flash', 'pos': [px, py], 'facing': self.player_facing, 'timer': 5})
            self.projectiles.append({'pos': list(self.player_pos), 'facing': self.player_facing})

        return enemies_defeated

    def _update_projectiles(self):
        projectiles_to_remove = []
        enemies_to_remove = []
        for proj in self.projectiles:
            # Move projectile
            if proj['facing'] == 1: proj['pos'][1] -= 1
            elif proj['facing'] == 2: proj['pos'][1] += 1
            elif proj['facing'] == 3: proj['pos'][0] -= 1
            elif proj['facing'] == 4: proj['pos'][0] += 1

            px, py = proj['pos']
            
            # Check out of bounds
            if not (0 <= px < self.GRID_W and 0 <= py < self.GRID_H):
                projectiles_to_remove.append(proj)
                continue

            # Check for enemy collision
            hit = False
            for enemy in self.enemies:
                if enemy['pos'] == proj['pos']:
                    enemy['health'] -= 1
                    enemy['damage_flash'] = 5
                    hit = True
                    if enemy['health'] <= 0 and enemy not in enemies_to_remove:
                        enemies_to_remove.append(enemy)
                        self.score += 1
                        self.effects.append({'type': 'explosion', 'pos': enemy['pos'], 'timer': 20, 'particles': []})
                        # sfx: explosion.wav
                    break
            
            if hit:
                projectiles_to_remove.append(proj)
        
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]

    def _update_enemies(self):
        player_pos_tuple = tuple(self.player_pos)
        enemy_positions = {tuple(e['pos']) for e in self.enemies}

        for enemy in self.enemies:
            if enemy['damage_flash'] > 0: enemy['damage_flash'] -= 1
            
            ex, ey = enemy['pos']
            
            # --- Enemy Attack ---
            if abs(ex - self.player_pos[0]) <= 1 and abs(ey - self.player_pos[1]) <= 1 and (ex, ey) != player_pos_tuple:
                self.player_health -= 1
                self.player_damage_flash = 10
                # sfx: player_hit.wav
                continue # Enemy attacks instead of moving
            
            # --- Enemy Movement ---
            possible_moves = []
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = ex + dx, ey + dy
                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H:
                    if (nx, ny) != player_pos_tuple and (nx, ny) not in enemy_positions:
                        possible_moves.append([nx, ny])
            
            if possible_moves:
                enemy_positions.remove(tuple(enemy['pos']))
                enemy['pos'] = random.choice(possible_moves)
                enemy_positions.add(tuple(enemy['pos']))

    def _update_effects(self):
        if self.player_damage_flash > 0: self.player_damage_flash -= 1
        
        effects_to_keep = []
        for effect in self.effects:
            effect['timer'] -= 1
            if effect['timer'] > 0:
                if effect['type'] == 'shockwave':
                    effect['radius'] += effect['max_radius'] / 10
                elif effect['type'] == 'explosion':
                    # Add new particles
                    for _ in range(3):
                        angle = self.np_random.uniform(0, 2 * math.pi)
                        speed = self.np_random.uniform(1, 4)
                        effect['particles'].append({
                            'offset': [0, 0],
                            'vel': [math.cos(angle) * speed, math.sin(angle) * speed]
                        })
                    # Update existing particles
                    for p in effect['particles']:
                        p['offset'][0] += p['vel'][0]
                        p['offset'][1] += p['vel'][1]
                effects_to_keep.append(effect)
        self.effects = effects_to_keep

    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            return True
        if not self.enemies:
            self.game_over = True
            self.win = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _grid_to_pixel(self, grid_pos):
        return (
            int(grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2),
            int(grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        )

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_W + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_H + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))

        # Draw projectiles
        for proj in self.projectiles:
            px, py = self._grid_to_pixel(proj['pos'])
            pygame.gfxdraw.filled_circle(self.screen, px, py, 4, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, px, py, 4, self.COLOR_PROJECTILE)

        # Draw enemies
        for enemy in self.enemies:
            px, py = self._grid_to_pixel(enemy['pos'])
            size = int(self.CELL_SIZE * 0.35)
            color = self.COLOR_ENEMY
            if enemy['damage_flash'] > 0:
                flash_alpha = (enemy['damage_flash'] / 5)
                color = (
                    int(self.COLOR_ENEMY[0] + (255 - self.COLOR_ENEMY[0]) * flash_alpha),
                    int(self.COLOR_ENEMY[1] + (255 - self.COLOR_ENEMY[1]) * flash_alpha),
                    int(self.COLOR_ENEMY[2] + (255 - self.COLOR_ENEMY[2]) * flash_alpha),
                )
            pygame.draw.rect(self.screen, color, (px - size, py - size, size * 2, size * 2))

        # Draw player
        px, py = self._grid_to_pixel(self.player_pos)
        size = int(self.CELL_SIZE * 0.4)
        color = self.COLOR_PLAYER
        if self.player_damage_flash > 0:
            flash_alpha = (self.player_damage_flash / 10)
            color = (
                int(self.COLOR_PLAYER[0] + (255 - self.COLOR_PLAYER[0]) * flash_alpha),
                int(self.COLOR_PLAYER[1] + (255 - self.COLOR_PLAYER[1]) * flash_alpha),
                int(self.COLOR_PLAYER[2] + (255 - self.COLOR_PLAYER[2]) * flash_alpha),
            )
        
        # Draw triangle based on facing direction
        points = []
        if self.player_facing == 1: # Up
            points = [(px, py - size), (px - size, py + size/2), (px + size, py + size/2)]
        elif self.player_facing == 2: # Down
            points = [(px, py + size), (px - size, py - size/2), (px + size, py - size/2)]
        elif self.player_facing == 3: # Left
            points = [(px - size, py), (px + size/2, py - size), (px + size/2, py + size)]
        elif self.player_facing == 4: # Right
            points = [(px + size, py), (px - size/2, py - size), (px - size/2, py + size)]
        
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Draw effects
        for effect in self.effects:
            pos_px, pos_py = self._grid_to_pixel(effect['pos'])
            if effect['type'] == 'shockwave':
                radius = int(effect['radius'] * self.CELL_SIZE)
                alpha = int(255 * (effect['timer'] / 10))
                if alpha > 0:
                    color = (*self.COLOR_SHOCKWAVE, alpha)
                    temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, color, (radius, radius), radius, width=max(1, int(radius/8)))
                    self.screen.blit(temp_surf, (pos_px - radius, pos_py - radius))
            elif effect['type'] == 'muzzle_flash':
                facing = effect['facing']
                offset_x, offset_y = 0, 0
                if facing == 1: offset_y = -self.CELL_SIZE/2
                elif facing == 2: offset_y = self.CELL_SIZE/2
                elif facing == 3: offset_x = -self.CELL_SIZE/2
                elif facing == 4: offset_x = self.CELL_SIZE/2
                flash_px, flash_py = pos_px + offset_x, pos_py + offset_y
                radius = int(self.CELL_SIZE * 0.2 * (effect['timer']/5))
                pygame.gfxdraw.filled_circle(self.screen, int(flash_px), int(flash_py), radius, self.COLOR_EXPLOSION_1)
            elif effect['type'] == 'explosion':
                for p in effect['particles']:
                    p_px = pos_px + p['offset'][0]
                    p_py = pos_py + p['offset'][1]
                    alpha = 255 * (effect['timer'] / 20)
                    color = self.COLOR_EXPLOSION_1 if self.np_random.random() > 0.3 else self.COLOR_EXPLOSION_2
                    pygame.draw.rect(self.screen, (*color, alpha), (p_px, p_py, 3, 3))


    def _render_ui(self):
        # Draw Health Bar
        health_bar_width = 150
        health_bar_height = 20
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, health_bar_width * health_ratio, health_bar_height))
        
        # Draw Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Draw Game Over
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.COLOR_ENEMY
            game_over_text = self.font_game_over.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "enemies_remaining": len(self.enemies),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Grid Combat Mech")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    
    # Game loop
    running = True
    while running:
        # Get user input
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

            # Since it's turn based, we only step if an action is taken or on a timer
            # For manual play, we step on any key press
            action_taken = any([movement, space_held, shift_held])
            
            # Allow waiting by pressing a dedicated wait key (e.g., 'w')
            if keys[pygame.K_w]:
                action_taken = True

            if action_taken:
                action = [movement, space_held, shift_held]
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")
        else: # If game is over, allow reset
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False


        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # In manual play, we control the speed.
        env.clock.tick(10) # Limit to 10 actions per second for playability

    env.close()