import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
from enum import Enum
import os
import pygame
pygame.init()


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.font.init()


class GameEnv(gym.Env):
    """
    A turn-based strategy game on a hexagonal grid where the player
    manipulates gravity wells to conquer enemy territory.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- User-facing metadata ---
    game_description = (
        "A turn-based strategy game where players control gravity-emitting particles to capture territory on a hexagonal grid from their opponent."
    )
    user_guide = (
        "Use arrow keys to navigate menus and aim. Press space to confirm selections and fire. Press shift to cancel or go back."
    )
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 75, 128)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (128, 25, 25)
    COLOR_NEUTRAL = (50, 60, 80)
    COLOR_GRID = (30, 35, 55)
    COLOR_TEXT = (220, 220, 240)
    COLOR_HIGHLIGHT = (255, 255, 0)
    COLOR_PROJECTILE = (255, 255, 200)

    # Screen and Grid Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    HEX_RADIUS = 22
    GRID_WIDTH = 13  # Number of hexes horizontally
    GRID_HEIGHT = 9  # Number of hexes vertically

    # Game Mechanics
    MAX_STEPS = 1000
    PARTICLE_MIN_SIZE = 5
    PARTICLE_MAX_SIZE = 15
    PARTICLE_SIZE_STEP = 1
    ENEMY_SPAWN_TURN_INTERVAL = 3

    class GamePhase(Enum):
        PLAYER_SELECT_PARTICLE = 1
        PLAYER_ADJUST_SIZE = 2
        PLAYER_AIM = 3
        PROJECTILE_FLIGHT = 4
        ENEMY_TURN = 5
        GAME_OVER = 6

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 48)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # --- Game State Variables ---
        self.hex_grid = {}
        self.player_particles = []
        self.enemy_particles = []
        self.projectiles = []
        self.steps = 0
        self.player_turns = 0
        self.score = 0
        self.phase = self.GamePhase.PLAYER_SELECT_PARTICLE
        self.selected_particle_idx = 0
        self.selected_target_idx = -1
        self.last_territory_count = 0
        
        self._initialize_hex_grid()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.player_turns = 0
        self.score = 0
        self.phase = self.GamePhase.PLAYER_SELECT_PARTICLE
        self.projectiles = []
        
        # Reset hex ownership
        for hex_tile in self.hex_grid.values():
            hex_tile['owner'] = None

        # Initialize particles
        self.player_particles = self._spawn_initial_particles('player')
        self.enemy_particles = self._spawn_initial_particles('enemy')
        
        self._update_territory()
        self.last_territory_count = self._count_territory('player')

        self.selected_particle_idx = 0 if self.player_particles else -1
        self.selected_target_idx = 0 if self.enemy_particles else -1
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False
        
        # --- Handle player input based on game phase ---
        if self.phase == self.GamePhase.PLAYER_SELECT_PARTICLE:
            if self.player_particles:
                if movement in [1, 4]: # Up or Right -> Next
                    self.selected_particle_idx = (self.selected_particle_idx + 1) % len(self.player_particles)
                elif movement in [2, 3]: # Down or Left -> Previous
                    self.selected_particle_idx = (self.selected_particle_idx - 1 + len(self.player_particles)) % len(self.player_particles)
                if space_held:
                    self.phase = self.GamePhase.PLAYER_ADJUST_SIZE

        elif self.phase == self.GamePhase.PLAYER_ADJUST_SIZE:
            if self.player_particles:
                particle = self.player_particles[self.selected_particle_idx]
                if movement == 1: # Up
                    particle['size'] = min(self.PARTICLE_MAX_SIZE, particle['size'] + self.PARTICLE_SIZE_STEP)
                elif movement == 2: # Down
                    particle['size'] = max(self.PARTICLE_MIN_SIZE, particle['size'] - self.PARTICLE_SIZE_STEP)
            
            if space_held:
                if self.enemy_particles:
                    self.selected_target_idx = 0
                    self.phase = self.GamePhase.PLAYER_AIM
                else: # No enemies to aim at, skip to end turn
                    self.phase = self.GamePhase.ENEMY_TURN
            if shift_held:
                self.phase = self.GamePhase.PLAYER_SELECT_PARTICLE

        elif self.phase == self.GamePhase.PLAYER_AIM:
            if self.enemy_particles:
                if movement in [1, 4]: # Up or Right -> Next
                    self.selected_target_idx = (self.selected_target_idx + 1) % len(self.enemy_particles)
                elif movement in [2, 3]: # Down or Left -> Previous
                    self.selected_target_idx = (self.selected_target_idx - 1 + len(self.enemy_particles)) % len(self.enemy_particles)

                if space_held:
                    if self.player_particles:
                        source_particle = self.player_particles[self.selected_particle_idx]
                        target_particle = self.enemy_particles[self.selected_target_idx]
                        self._create_projectile(source_particle, target_particle)
                        self.phase = self.GamePhase.PROJECTILE_FLIGHT
            else: # Should not happen if check is done before, but as a safeguard
                self.phase = self.GamePhase.ENEMY_TURN

            if shift_held:
                self.phase = self.GamePhase.PLAYER_ADJUST_SIZE

        # --- Handle automatic game phases ---
        if self.phase == self.GamePhase.PROJECTILE_FLIGHT:
            proj_reward, proj_terminated = self._update_projectiles()
            reward += proj_reward
            terminated = terminated or proj_terminated
            if not self.projectiles: # All projectiles finished
                self._update_territory()
                
                current_territory_count = self._count_territory('player')
                reward += (current_territory_count - self.last_territory_count) * 0.1
                self.last_territory_count = current_territory_count

                if not terminated:
                    self.phase = self.GamePhase.ENEMY_TURN
                    self.player_turns += 1

        if self.phase == self.GamePhase.ENEMY_TURN:
            if not terminated:
                enemy_reward, enemy_terminated = self._execute_enemy_turn()
                reward += enemy_reward
                terminated = terminated or enemy_terminated
                
                self._update_territory()
                current_territory_count = self._count_territory('player')
                reward += (current_territory_count - self.last_territory_count) * 0.1
                self.last_territory_count = current_territory_count
                
                if not terminated:
                    if self.player_turns > 0 and self.player_turns % self.ENEMY_SPAWN_TURN_INTERVAL == 0:
                        self._spawn_new_enemy()
                    
                    self.phase = self.GamePhase.PLAYER_SELECT_PARTICLE
                    if not self.player_particles:
                        terminated = True
                        reward -= 100.0 # Loss
                        self.phase = self.GamePhase.GAME_OVER

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            player_territory = self._count_territory('player')
            enemy_territory = self._count_territory('enemy')
            reward += (player_territory - enemy_territory) * 0.1
        
        if terminated:
            self.phase = self.GamePhase.GAME_OVER
        
        return self._get_observation(), float(reward), terminated, False, self._get_info()

    def _execute_enemy_turn(self):
        turn_reward = 0.0
        terminated = False
        
        particles_to_remove = []
        for enemy in self.enemy_particles:
            if not self.player_particles:
                break
            
            closest_player = min(self.player_particles, key=lambda p: self._hex_distance(enemy['q'], enemy['r'], p['q'], p['r']))
            
            damage = enemy['size']
            closest_player['size'] -= damage
            
            if closest_player['size'] <= 0:
                if closest_player not in particles_to_remove:
                    particles_to_remove.append(closest_player)
        
        for p in particles_to_remove:
            if p in self.player_particles:
                self.player_particles.remove(p)

        if not self.player_particles:
            terminated = True
            turn_reward -= 100.0
        
        return turn_reward, terminated

    def _create_projectile(self, source, target):
        start_pos = self._axial_to_pixel(source['q'], source['r'])
        end_pos = self._axial_to_pixel(target['q'], target['r'])
        self.projectiles.append({
            'pos': np.array(start_pos, dtype=float),
            'target_pos': np.array(end_pos, dtype=float),
            'target_particle': target,
            'source_particle': source,
            'speed': 15.0,
        })
    
    def _update_projectiles(self):
        reward = 0.0
        terminated = False
        
        for proj in self.projectiles[:]:
            direction = proj['target_pos'] - proj['pos']
            distance = np.linalg.norm(direction)
            
            if distance < proj['speed']:
                target = proj['target_particle']
                source = proj['source_particle']
                
                damage = source['size']
                target['size'] -= damage

                if target['size'] <= 0:
                    if target in self.enemy_particles:
                        self.enemy_particles.remove(target)
                        reward += 1.0 # Reward for destroying enemy
                    if not self.enemy_particles:
                        terminated = True
                        reward += 100.0 # Win
                
                self.projectiles.remove(proj)
            else:
                proj['pos'] += (direction / distance) * proj['speed']
        
        return reward, terminated

    def _update_territory(self):
        if not self.player_particles and not self.enemy_particles:
            for hex_tile in self.hex_grid.values():
                hex_tile['owner'] = None
            return

        for hex_coord, hex_tile in self.hex_grid.items():
            q, r = hex_coord
            player_influence = 0.0
            enemy_influence = 0.0

            for p in self.player_particles:
                dist = self._hex_distance(q, r, p['q'], p['r'])
                player_influence += p['size'] / max(1, dist**2)

            for e in self.enemy_particles:
                dist = self._hex_distance(q, r, e['q'], e['r'])
                enemy_influence += e['size'] / max(1, dist**2)
            
            if player_influence > enemy_influence * 1.1:
                hex_tile['owner'] = 'player'
            elif enemy_influence > player_influence * 1.1:
                hex_tile['owner'] = 'enemy'
            else:
                hex_tile['owner'] = None

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
            "player_particles": len(self.player_particles),
            "enemy_particles": len(self.enemy_particles),
            "territory_player": self._count_territory('player'),
            "territory_enemy": self._count_territory('enemy'),
        }

    def _render_game(self):
        for hex_coord, hex_tile in self.hex_grid.items():
            center_pos = self._axial_to_pixel(hex_coord[0], hex_coord[1])
            if hex_tile['owner'] == 'player':
                color = self.COLOR_PLAYER_GLOW
            elif hex_tile['owner'] == 'enemy':
                color = self.COLOR_ENEMY_GLOW
            else:
                color = self.COLOR_GRID
            self._draw_hexagon(self.screen, color, center_pos, self.HEX_RADIUS, filled=True)
            self._draw_hexagon(self.screen, self.COLOR_GRID, center_pos, self.HEX_RADIUS, width=1)

        all_particles = self.player_particles + self.enemy_particles
        for p in all_particles:
            pos = self._axial_to_pixel(p['q'], p['r'])
            color = self.COLOR_PLAYER if p['owner'] == 'player' else self.COLOR_ENEMY
            glow_color = self.COLOR_PLAYER_GLOW if p['owner'] == 'player' else self.COLOR_ENEMY_GLOW
            self._draw_glowing_circle(self.screen, color, glow_color, pos, p['size'])

        if self.phase in [self.GamePhase.PLAYER_SELECT_PARTICLE, self.GamePhase.PLAYER_ADJUST_SIZE, self.GamePhase.PLAYER_AIM] and self.player_particles:
            p = self.player_particles[self.selected_particle_idx]
            pos = self._axial_to_pixel(p['q'], p['r'])
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.PARTICLE_MAX_SIZE + 3, self.COLOR_HIGHLIGHT)

        if self.phase == self.GamePhase.PLAYER_AIM and self.enemy_particles and self.player_particles:
            e = self.enemy_particles[self.selected_target_idx]
            pos = self._axial_to_pixel(e['q'], e['r'])
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.PARTICLE_MAX_SIZE + 3, self.COLOR_HIGHLIGHT)
            
            source_pos = self._axial_to_pixel(self.player_particles[self.selected_particle_idx]['q'], self.player_particles[self.selected_particle_idx]['r'])
            pygame.draw.aaline(self.screen, self.COLOR_HIGHLIGHT, source_pos, pos)

        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            self._draw_glowing_circle(self.screen, self.COLOR_PROJECTILE, self.COLOR_HIGHLIGHT, pos, 4)

    def _render_ui(self):
        phase_str = self.phase.name.replace("_", " ").title()
        phase_text = self.font_medium.render(phase_str, True, self.COLOR_TEXT)
        self.screen.blit(phase_text, (self.SCREEN_WIDTH // 2 - phase_text.get_width() // 2, 10))

        player_territory = self._count_territory('player')
        enemy_territory = self._count_territory('enemy')
        total_territory = len(self.hex_grid)
        player_perc = (player_territory / total_territory) * 100 if total_territory > 0 else 0
        enemy_perc = (enemy_territory / total_territory) * 100 if total_territory > 0 else 0
        
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        territory_text_player = self.font_small.render(f"PLAYER: {player_perc:.1f}%", True, self.COLOR_PLAYER)
        self.screen.blit(territory_text_player, (10, self.SCREEN_HEIGHT - 30))
        
        territory_text_enemy = self.font_small.render(f"ENEMY: {enemy_perc:.1f}%", True, self.COLOR_ENEMY)
        self.screen.blit(territory_text_enemy, (self.SCREEN_WIDTH - territory_text_enemy.get_width() - 10, self.SCREEN_HEIGHT - 30))
        
        if self.phase == self.GamePhase.GAME_OVER:
            win_txt = "VICTORY" if len(self.enemy_particles) == 0 and len(self.player_particles) > 0 else "DEFEAT"
            color = self.COLOR_PLAYER if win_txt == "VICTORY" else self.COLOR_ENEMY
            end_text = self.font_large.render(win_txt, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH // 2 - end_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - end_text.get_height() // 2))

    def _initialize_hex_grid(self):
        self.hex_grid = {}
        for q in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT):
                q_offset = q - r // 2
                if 0 <= q_offset < self.GRID_WIDTH and 0 <= r < self.GRID_HEIGHT:
                    center = self._axial_to_pixel(q_offset, r)
                    if 0 < center[0] < self.SCREEN_WIDTH and 0 < center[1] < self.SCREEN_HEIGHT:
                        self.hex_grid[(q_offset, r)] = {'owner': None}

    def _spawn_initial_particles(self, owner):
        particles = []
        num_particles = 2 if owner == 'player' else 3
        
        for _ in range(num_particles):
            while True:
                q = self.np_random.integers(0, self.GRID_WIDTH) - self.GRID_WIDTH // 2
                r = self.np_random.integers(0, self.GRID_HEIGHT)
                
                y_pos_norm = r / self.GRID_HEIGHT
                is_player_zone = y_pos_norm > 0.6
                is_enemy_zone = y_pos_norm < 0.4
                
                if (owner == 'player' and is_player_zone) or \
                   (owner == 'enemy' and is_enemy_zone):
                    if (q, r) in self.hex_grid and not self._is_hex_occupied(q, r):
                        break
            
            particles.append({'q': q, 'r': r, 'size': 10, 'owner': owner})
        return particles
    
    def _spawn_new_enemy(self):
        enemy_hexes = [coord for coord, tile in self.hex_grid.items() if tile['owner'] == 'enemy' and not self._is_hex_occupied(coord[0], coord[1])]
        if not enemy_hexes:
            enemy_hexes = [coord for coord, tile in self.hex_grid.items() if tile['owner'] is None and not self._is_hex_occupied(coord[0], coord[1])]
        if not enemy_hexes:
            return

        idx = self.np_random.integers(len(enemy_hexes))
        q, r = enemy_hexes[idx]
        self.enemy_particles.append({'q': q, 'r': r, 'size': 10, 'owner': 'enemy'})

    def _is_hex_occupied(self, q, r):
        for p in self.player_particles + self.enemy_particles:
            if p['q'] == q and p['r'] == r:
                return True
        return False

    def _count_territory(self, owner):
        return sum(1 for tile in self.hex_grid.values() if tile['owner'] == owner)

    def _axial_to_pixel(self, q, r):
        x = self.HEX_RADIUS * (3/2 * q) + self.SCREEN_WIDTH / 2
        y = self.HEX_RADIUS * (math.sqrt(3)/2 * q + math.sqrt(3) * r) + self.SCREEN_HEIGHT / 2 - (self.GRID_HEIGHT/2 * self.HEX_RADIUS * math.sqrt(3))
        return (x, y)

    def _hex_distance(self, q1, r1, q2, r2):
        s1 = -q1 - r1
        s2 = -q2 - r2
        return (abs(q1 - q2) + abs(r1 - r2) + abs(s1 - s2)) / 2

    def _draw_hexagon(self, surface, color, center, radius, width=0, filled=False):
        points = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.pi / 180 * angle_deg
            points.append((center[0] + radius * math.cos(angle_rad),
                           center[1] + radius * math.sin(angle_rad)))
        if filled:
            pygame.gfxdraw.filled_polygon(surface, points, color)
        else:
            pygame.gfxdraw.aapolygon(surface, points, color)

    def _draw_glowing_circle(self, surface, color, glow_color, center, radius):
        center_int = (int(center[0]), int(center[1]))
        radius = int(radius)
        if radius <= 0: return

        for i in range(radius // 2, 0, -1):
            alpha = 255 // (i * 2)
            if len(glow_color) == 3:
                temp_color = (*glow_color, alpha)
            else:
                temp_color = (glow_color[0], glow_color[1], glow_color[2], alpha)
            
            # Pygame < 2.0.0 doesn't handle alpha in filled_circle well, but it's what we have
            s = pygame.Surface((2 * (radius + i), 2 * (radius + i)), pygame.SRCALPHA)
            pygame.draw.circle(s, temp_color, (radius + i, radius + i), radius + i)
            surface.blit(s, (center_int[0] - radius - i, center_int[1] - radius - i))

        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius, color)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    env = GameEnv()
    
    pygame.display.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Particle Gravity Siege")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("----------------\n")
    
    while not done:
        action = np.array([0, 0, 0]) 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                
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

        if action.any():
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Phase: {env.phase.name}, Reward: {reward:.2f}, Terminated: {terminated}")
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']:.2f}")
                # Wait for a moment before closing
                pygame.time.wait(2000)
                done = True

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()