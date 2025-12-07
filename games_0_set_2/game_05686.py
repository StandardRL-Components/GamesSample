
# Generated: 2025-08-28T05:45:15.295486
# Source Brief: brief_05686.md
# Brief Index: 5686

        
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
        "Controls: Arrow keys to move. Space to shoot in your last moved direction. Survive the horde!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Top-down, grid-based zombie survival. Clear 5 waves of zombies by moving strategically and managing your health and ammo."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_W, GRID_H = 16, 10
    CELL_SIZE = 40
    SCREEN_W, SCREEN_H = GRID_W * CELL_SIZE, GRID_H * CELL_SIZE

    COLOR_BG = (20, 20, 25)
    COLOR_GRID = (40, 40, 50)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_ZOMBIE = (220, 50, 50)
    COLOR_HEALTH_PACK = (50, 150, 255)
    COLOR_PROJECTILE = (255, 255, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_UI_BG = (30, 30, 40, 180)

    MAX_STEPS = 1000
    TOTAL_WAVES = 5
    ZOMBIE_SPAWN_START = 20
    ZOMBIE_SPAWN_INCREMENT = 2
    ZOMBIE_AGGRO_RADIUS = 5
    ZOMBIE_DAMAGE = 10
    ZOMBIE_HEALTH = 20
    PROJECTILE_DAMAGE = 20
    HEALTH_PACK_HEAL = 25
    MAX_HEALTH_PACKS = 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # RNG
        self.np_random = None

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_ammo = None
        self.player_facing = None
        self.player_flash_timer = 0
        self.zombies = []
        self.health_packs = []
        self.projectiles = []
        self.particles = []
        self.wave = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()


        self.player_pos = (self.GRID_W // 2, self.GRID_H // 2)
        self.player_health = 100
        self.player_ammo = 50
        self.player_facing = (0, -1)  # Default up
        self.player_flash_timer = 0

        self.zombies = []
        self.health_packs = []
        self.projectiles = []
        self.particles = []

        self.wave = 1
        self._spawn_zombies()
        self._spawn_health_packs()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.1  # Survival penalty
        self.steps += 1
        self.player_flash_timer = max(0, self.player_flash_timer - 1)

        # 1. Unpack action and handle player turn
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_player_action(movement, space_held)

        # 2. Update game entities
        reward += self._update_projectiles()
        self._update_zombies()
        self._update_particles()

        # 3. Handle collisions and pickups
        reward += self._handle_collisions()

        # 4. Check for wave completion
        if not self.zombies and self.wave <= self.TOTAL_WAVES:
            if self.wave == self.TOTAL_WAVES:
                self.game_won = True
                reward += 100
            else:
                self.wave += 1
                reward += 20
                self._spawn_zombies()
                self._spawn_health_packs()
                # SFX: Wave Complete
        
        self.score += reward
        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS or self.game_won
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_player_action(self, movement, space_held):
        # Movement
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            self.player_facing = (dx, dy)
            new_x = self.player_pos[0] + dx
            new_y = self.player_pos[1] + dy
            if 0 <= new_x < self.GRID_W and 0 <= new_y < self.GRID_H:
                self.player_pos = (new_x, new_y)

        # Shooting
        if space_held and self.player_ammo > 0:
            self.player_ammo -= 1
            self.projectiles.append({
                "pos": list(self.player_pos),
                "dir": self.player_facing
            })
            # Muzzle flash
            self._create_particles(self.player_pos, self.COLOR_PROJECTILE, 5, 2)
            # SFX: Laser Shot

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for p in self.projectiles:
            p['pos'][0] += p['dir'][0]
            p['pos'][1] += p['dir'][1]
            px, py = p['pos']

            hit_zombie = False
            for z in self.zombies:
                if z['pos'] == (int(px), int(py)):
                    z['health'] -= self.PROJECTILE_DAMAGE
                    reward += 0.5
                    hit_zombie = True
                    self._create_particles(z['pos'], self.COLOR_ZOMBIE, 10, 3)
                    # SFX: Zombie Hit
                    break
            
            if not (0 <= px < self.GRID_W and 0 <= py < self.GRID_H) or hit_zombie:
                continue # Projectile is destroyed
            
            projectiles_to_keep.append(p)

        self.projectiles = projectiles_to_keep
        return reward

    def _update_zombies(self):
        occupied_cells = {z['pos'] for z in self.zombies}
        for z in self.zombies:
            dist_x = self.player_pos[0] - z['pos'][0]
            dist_y = self.player_pos[1] - z['pos'][1]
            
            if abs(dist_x) + abs(dist_y) <= self.ZOMBIE_AGGRO_RADIUS:
                move_x, move_y = np.sign(dist_x), np.sign(dist_y)
                
                # Try moving horizontally first
                if move_x != 0:
                    new_pos = (z['pos'][0] + move_x, z['pos'][1])
                    if new_pos not in occupied_cells:
                        occupied_cells.remove(z['pos'])
                        z['pos'] = new_pos
                        occupied_cells.add(new_pos)
                        continue
                
                # Then try vertically
                if move_y != 0:
                    new_pos = (z['pos'][0], z['pos'][1] + move_y)
                    if new_pos not in occupied_cells:
                        occupied_cells.remove(z['pos'])
                        z['pos'] = new_pos
                        occupied_cells.add(new_pos)

    def _handle_collisions(self):
        reward = 0
        # Player vs Zombies
        zombies_at_player = [z for z in self.zombies if z['pos'] == self.player_pos]
        if zombies_at_player:
            self.player_health -= self.ZOMBIE_DAMAGE * len(zombies_at_player)
            self.player_health = max(0, self.player_health)
            self.player_flash_timer = 3 # Flash for 3 frames
            # SFX: Player Hurt
        
        # Player vs Health Packs
        packs_to_keep = []
        for pack_pos in self.health_packs:
            if pack_pos == self.player_pos:
                self.player_health = min(100, self.player_health + self.HEALTH_PACK_HEAL)
                reward += 1.0
                self._create_particles(self.player_pos, self.COLOR_HEALTH_PACK, 15, 4)
                # SFX: Health Pickup
            else:
                packs_to_keep.append(pack_pos)
        self.health_packs = packs_to_keep

        # Zombie deaths
        zombies_to_keep = []
        for z in self.zombies:
            if z['health'] <= 0:
                reward += 10.0
                self._create_particles(z['pos'], self.COLOR_ZOMBIE, 20, 5, is_death=True)
                # SFX: Zombie Death
            else:
                zombies_to_keep.append(z)
        self.zombies = zombies_to_keep

        return reward

    def _spawn_zombies(self):
        num_zombies = self.ZOMBIE_SPAWN_START + (self.wave - 1) * self.ZOMBIE_SPAWN_INCREMENT
        occupied = {z['pos'] for z in self.zombies} | {self.player_pos}
        for _ in range(num_zombies):
            while True:
                edge = self.np_random.integers(4)
                if edge == 0: x, y = 0, self.np_random.integers(self.GRID_H)
                elif edge == 1: x, y = self.GRID_W - 1, self.np_random.integers(self.GRID_H)
                elif edge == 2: x, y = self.np_random.integers(self.GRID_W), 0
                else: x, y = self.np_random.integers(self.GRID_W), self.GRID_H - 1
                
                if (x, y) not in occupied:
                    self.zombies.append({'pos': (x, y), 'health': self.ZOMBIE_HEALTH})
                    occupied.add((x, y))
                    break

    def _spawn_health_packs(self):
        occupied = {z['pos'] for z in self.zombies} | {hp for hp in self.health_packs} | {self.player_pos}
        while len(self.health_packs) < self.MAX_HEALTH_PACKS:
            x, y = self.np_random.integers(self.GRID_W), self.np_random.integers(self.GRID_H)
            if (x, y) not in occupied:
                self.health_packs.append((x, y))
                occupied.add((x, y))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Grid
        for x in range(self.GRID_W):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x * self.CELL_SIZE, 0), (x * self.CELL_SIZE, self.SCREEN_H))
        for y in range(self.GRID_H):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y * self.CELL_SIZE), (self.SCREEN_W, y * self.CELL_SIZE))

        # Health Packs
        pulse = abs(math.sin(self.steps * 0.1)) * 5
        for hx, hy in self.health_packs:
            cx, cy = int((hx + 0.5) * self.CELL_SIZE), int((hy + 0.5) * self.CELL_SIZE)
            r = int(self.CELL_SIZE * 0.3 + pulse)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, r, self.COLOR_HEALTH_PACK)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, r, self.COLOR_HEALTH_PACK)

        # Projectiles
        for p in self.projectiles:
            px, py = (p['pos'][0] + 0.5) * self.CELL_SIZE, (p['pos'][1] + 0.5) * self.CELL_SIZE
            l = self.CELL_SIZE * 0.4
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, 
                             (int(px - p['dir'][0]*l), int(py - p['dir'][1]*l)),
                             (int(px + p['dir'][0]*l), int(py + p['dir'][1]*l)), 4)
        
        # Zombies
        for z in self.zombies:
            zx, zy = (z['pos'][0] + 0.5) * self.CELL_SIZE, (z['pos'][1] + 0.5) * self.CELL_SIZE
            size = int(self.CELL_SIZE * 0.35)
            rect = pygame.Rect(int(zx - size), int(zy - size), size * 2, size * 2)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, rect, border_radius=4)
        
        # Player
        px, py = (self.player_pos[0] + 0.5) * self.CELL_SIZE, (self.player_pos[1] + 0.5) * self.CELL_SIZE
        size = int(self.CELL_SIZE * 0.4)
        color = (255, 255, 255) if self.player_flash_timer > 0 else self.COLOR_PLAYER
        
        player_rect = pygame.Rect(int(px - size), int(py - size), size * 2, size * 2)
        pygame.draw.rect(self.screen, color, player_rect, border_radius=6)
        
        # Facing indicator
        fx, fy = self.player_facing
        indicator_start = (px, py)
        indicator_end = (px + fx * size * 0.8, py + fy * size * 0.8)
        pygame.draw.line(self.screen, self.COLOR_BG, indicator_start, indicator_end, 5)

        # Particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.circle(self.screen, p['color'], pos, int(p['size']))

    def _render_ui(self):
        # UI Background
        ui_panel = pygame.Surface((self.SCREEN_W, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # Health Bar
        health_ratio = self.player_health / 100
        health_color = (int(220 * (1 - health_ratio)), int(220 * health_ratio), 40)
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, health_color, (10, 10, 200 * health_ratio, 20))
        health_text = self.font_small.render(f"HP: {self.player_health}/100", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (220, 12))

        # Ammo
        ammo_text = self.font_small.render(f"AMMO: {self.player_ammo}", True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (350, 12))

        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.wave}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (480, 12))

        # Game Over / Win Text
        if self.player_health <= 0:
            self._render_centered_text("GAME OVER", self.COLOR_ZOMBIE)
        elif self.game_won:
            self._render_centered_text("YOU SURVIVED!", self.COLOR_PLAYER)

    def _render_centered_text(self, text, color):
        text_surf = self.font_large.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_W / 2, self.SCREEN_H / 2))
        
        bg_rect = text_rect.inflate(20, 20)
        bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, 180))
        self.screen.blit(bg_surf, bg_rect)
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, pos, color, count, speed_max, is_death=False):
        cx, cy = (pos[0] + 0.5) * self.CELL_SIZE, (pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = self.np_random.uniform(2, 5) if not is_death else self.np_random.uniform(4, 10)
            lifespan = self.np_random.integers(10, 20)
            self.particles.append({'pos': [cx, cy], 'vel': vel, 'size': size, 'color': color, 'life': lifespan})

    def _update_particles(self):
        particles_to_keep = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['size'] -= 0.2
            p['life'] -= 1
            if p['size'] > 0 and p['life'] > 0:
                particles_to_keep.append(p)
        self.particles = particles_to_keep

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "wave": self.wave,
            "zombies_left": len(self.zombies),
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
        assert test_obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_window = pygame.display.set_mode((GameEnv.SCREEN_W, GameEnv.SCREEN_H))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        # --- Human Input ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already the rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        game_window.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")
            pygame.time.wait(3000) # Wait 3 seconds
            obs, info = env.reset()

        clock.tick(10) # Control the speed of the manual game

    env.close()