import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:19:09.130634
# Source Brief: brief_00397.md
# Brief Index: 397
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game objects
class Particle:
    def __init__(self, x, y, color, life, size, angle, speed):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.initial_life = life
        self.size = size
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def update(self):
        self.life -= 1
        self.x += self.vx
        self.y += self.vy
        self.size = max(0, self.size * 0.95)
        self.vx *= 0.98
        self.vy *= 0.98

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.initial_life))
            color_with_alpha = self.color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.size), color_with_alpha)

class Enemy:
    def __init__(self, path, speed=1.5, health=100):
        self.path = path
        self.pos = list(self.path[0])
        self.target_waypoint_idx = 1
        self.max_health = health
        self.health = health
        self.speed_multiplier = 1.0
        self.base_speed = speed
        self.stun_timer = 0
        self.is_alive = True
        self.size = 12

    def update(self):
        if not self.is_alive or self.target_waypoint_idx >= len(self.path):
            return

        if self.stun_timer > 0:
            self.stun_timer -= 1
            return

        target_pos = self.path[self.target_waypoint_idx]
        direction = [target_pos[0] - self.pos[0], target_pos[1] - self.pos[1]]
        distance = math.hypot(*direction)
        
        speed = self.base_speed * self.speed_multiplier
        if distance < speed:
            self.pos = list(target_pos)
            self.target_waypoint_idx += 1
            if self.target_waypoint_idx >= len(self.path):
                self.is_alive = False # Reached the end
        else:
            direction_norm = [d / distance for d in direction]
            self.pos[0] += direction_norm[0] * speed
            self.pos[1] += direction_norm[1] * speed
        
        # Reset speed multiplier each frame, traps will re-apply it
        self.speed_multiplier = 1.0

    def draw(self, surface):
        if not self.is_alive:
            return
        
        # Body
        rect = pygame.Rect(int(self.pos[0] - self.size/2), int(self.pos[1] - self.size/2), self.size, self.size)
        pygame.draw.rect(surface, (220, 50, 50), rect, border_radius=2)
        
        # Health bar
        if self.health < self.max_health:
            health_pct = self.health / self.max_health
            bar_width = self.size * 1.2
            bar_height = 4
            bar_x = self.pos[0] - bar_width / 2
            bar_y = self.pos[1] - self.size - 4
            
            pygame.draw.rect(surface, (80, 20, 20), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(surface, (50, 200, 50), (bar_x, bar_y, bar_width * health_pct, bar_height))

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.is_alive = False
            return True # Died
        return False

class Trap:
    TYPES = ['slow', 'stun', 'kill']
    COLORS = {
        'slow': (255, 100, 0),
        'stun': (0, 150, 255),
        'kill': (120, 0, 150)
    }
    
    def __init__(self, pos):
        self.pos = pos
        self.type_idx = 0
        self.type = self.TYPES[self.type_idx]
        self.color = self.COLORS[self.type]
        self.radius = 25
        self.activation_prob = 0.20
        self.cooldown = 0
        self.max_cooldown = 30 # 1 second at 30fps
        self.activation_effect_timer = 0

    def cycle_type(self):
        self.type_idx = (self.type_idx + 1) % len(self.TYPES)
        self.type = self.TYPES[self.type_idx]
        self.color = self.COLORS[self.type]
        # SFX: cycle_type

    def update(self):
        if self.cooldown > 0:
            self.cooldown -= 1
        if self.activation_effect_timer > 0:
            self.activation_effect_timer -= 1

    def activate(self, enemy, particles):
        if self.cooldown > 0:
            return 0.0

        if random.random() < self.activation_prob:
            self.cooldown = self.max_cooldown
            self.activation_effect_timer = 10 # visual effect duration
            
            for _ in range(15):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 4)
                size = random.uniform(2, 5)
                life = random.randint(10, 20)
                particles.append(Particle(self.pos[0], self.pos[1], self.color, life, size, angle, speed))

            if self.type == 'slow':
                enemy.speed_multiplier = 0.3
                # SFX: trap_fire_slow
                return 0.1
            elif self.type == 'stun':
                enemy.stun_timer = 60 # 2 seconds
                # SFX: trap_fire_stun
                return 0.2
            elif self.type == 'kill':
                if enemy.take_damage(enemy.max_health): # Instant kill
                    # SFX: trap_fire_kill + enemy_die
                    return 1.0
        return 0.0

    def draw(self, surface):
        # Inner circle indicating type
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), 8, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), 8, self.color)
        
        # Cooldown indicator
        if self.cooldown > 0:
            alpha = int(150 * (self.cooldown / self.max_cooldown))
            pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), self.radius, (50, 50, 60, alpha))

        # Activation effect
        if self.activation_effect_timer > 0:
            progress = self.activation_effect_timer / 10
            radius = int(self.radius * (1.5 - progress * 0.5))
            alpha = int(200 * progress)
            pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), radius, self.color + (alpha,))
            pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), radius - 2, self.color + (alpha,))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A strategic tower defense game. Place and cycle through different traps to stop "
        "waves of enemies from reaching the end of the path."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press Shift to place a new trap "
        "and Space to cycle its type."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2500
        self.TOTAL_ENEMIES_TO_DEFEAT = 100
        self.ENEMY_SPAWN_RATE = 45 # Lower is faster

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_PATH = (70, 80, 90)
        self.COLOR_TRAP_SLOT = (50, 60, 70)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Consolas', 24)
        
        # Game world setup
        self._define_world()
        
        # Initialize state variables (done in reset)
        self.enemies = []
        self.traps = {}
        self.particles = []
        self.cursor_pos = (0, 0)
        self.prev_space_held = False
        self.prev_shift_held = False

        # self.reset() is called by the wrapper
    
    def _define_world(self):
        # Path for enemies
        self.PATH_WAYPOINTS = [
            (-20, 100), (100, 100), (100, 300), (300, 300),
            (300, 50), (540, 50), (540, 350), (self.WIDTH + 20, 350)
        ]

        # Grid of trap locations for easy navigation
        self.trap_grid = []
        self.trap_locations = {} # Map from (x,y) to trap object
        rows, cols = 4, 6
        x_spacing = self.WIDTH / (cols + 1)
        y_spacing = self.HEIGHT / (rows + 1)
        for r in range(rows):
            row_list = []
            for c in range(cols):
                x = int((c + 1) * x_spacing)
                y = int((r + 1) * y_spacing)
                # Jitter positions slightly for a more organic look
                x += random.randint(-5, 5)
                y += random.randint(-5, 5)
                pos = (x, y)
                row_list.append(pos)
                self.trap_locations[pos] = None
            self.trap_grid.append(row_list)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.enemies_defeated = 0
        self.enemies_spawned = 0
        self.spawn_timer = 0
        
        self.enemies = []
        self.traps = {}
        self.particles = []
        
        self.cursor_pos = (0, 0) # (row, col) in trap_grid
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        reward = 0.0

        # --- 1. Handle Player Input ---
        self._handle_input(movement, space_held, shift_held)

        # --- 2. Update Game State ---
        # Spawn new enemies
        self.spawn_timer += 1
        if self.spawn_timer >= self.ENEMY_SPAWN_RATE and self.enemies_spawned < self.TOTAL_ENEMIES_TO_DEFEAT:
            self.enemies.append(Enemy(self.PATH_WAYPOINTS))
            self.enemies_spawned += 1
            self.spawn_timer = 0
        
        # Update traps
        for trap in self.traps.values():
            trap.update()

        # Update enemies and check for trap interactions
        surviving_enemies = []
        for enemy in self.enemies:
            enemy.update()
            if enemy.is_alive:
                for trap in self.traps.values():
                    dist = math.hypot(enemy.pos[0] - trap.pos[0], enemy.pos[1] - trap.pos[1])
                    if dist < trap.radius:
                        trap_reward = trap.activate(enemy, self.particles)
                        reward += trap_reward
                        if not enemy.is_alive: # Killed by trap
                            self.enemies_defeated += 1
                            break # No need to check other traps for this dead enemy
                
                if enemy.is_alive:
                    surviving_enemies.append(enemy)
                else: # Died this frame (either by trap or reaching end)
                    if enemy.target_waypoint_idx < len(enemy.path): # Died by trap, not by finishing
                        self._create_death_particles(enemy.pos)
                        # SFX: enemy_die
            
        self.enemies = surviving_enemies

        # Update particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

        # --- 3. Calculate Reward & Termination ---
        self.score += reward
        terminated = (self.enemies_defeated >= self.TOTAL_ENEMIES_TO_DEFEAT)
        truncated = (self.steps >= self.MAX_STEPS)

        if self.enemies_defeated >= self.TOTAL_ENEMIES_TO_DEFEAT and not self.game_over:
            reward += 100.0
            self.score += 100.0
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Cursor movement
        r, c = self.cursor_pos
        if movement == 1: r = max(0, r - 1) # Up
        elif movement == 2: r = min(len(self.trap_grid) - 1, r + 1) # Down
        elif movement == 3: c = max(0, c - 1) # Left
        elif movement == 4: c = min(len(self.trap_grid[0]) - 1, c + 1) # Right
        self.cursor_pos = (r, c)

        space_just_pressed = space_held and not self.prev_space_held
        shift_just_pressed = shift_held and not self.prev_shift_held
        
        cursor_loc = self.trap_grid[self.cursor_pos[0]][self.cursor_pos[1]]

        # Place/create trap with Shift
        if shift_just_pressed:
            if cursor_loc not in self.traps:
                self.traps[cursor_loc] = Trap(cursor_loc)
                # SFX: place_trap
                self._create_placement_particles(cursor_loc)

        # Cycle trap type with Space
        if space_just_pressed and cursor_loc in self.traps:
            self.traps[cursor_loc].cycle_type()
            # SFX: cycle_type

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _create_death_particles(self, pos):
        for _ in range(25):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            size = random.uniform(2, 6)
            life = random.randint(20, 40)
            self.particles.append(Particle(pos[0], pos[1], (220, 50, 50), life, size, angle, speed))
    
    def _create_placement_particles(self, pos):
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2)
            size = random.uniform(1, 3)
            life = random.randint(10, 15)
            self.particles.append(Particle(pos[0], pos[1], self.COLOR_CURSOR, life, size, angle, speed))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        pygame.draw.aalines(self.screen, self.COLOR_PATH, False, self.PATH_WAYPOINTS, 3)

        # Draw trap slots
        for pos in self.trap_locations.keys():
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 20, self.COLOR_TRAP_SLOT)

        # Draw placed traps
        for trap in self.traps.values():
            trap.draw(self.screen)

        # Draw cursor
        cursor_loc = self.trap_grid[self.cursor_pos[0]][self.cursor_pos[1]]
        pulse = abs(math.sin(self.steps * 0.1))
        radius = int(25 + pulse * 4)
        alpha = int(150 + pulse * 105)
        pygame.gfxdraw.aacircle(self.screen, cursor_loc[0], cursor_loc[1], radius, self.COLOR_CURSOR + (alpha,))
        pygame.gfxdraw.aacircle(self.screen, cursor_loc[0], cursor_loc[1], radius - 1, self.COLOR_CURSOR + (alpha,))

        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(self.screen)
            if enemy.stun_timer > 0:
                pulse_size = int(8 + abs(math.sin(self.steps * 0.3)) * 4)
                pygame.gfxdraw.aacircle(self.screen, int(enemy.pos[0]), int(enemy.pos[1]), pulse_size, Trap.COLORS['stun'])

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        enemies_remaining = self.TOTAL_ENEMIES_TO_DEFEAT - self.enemies_defeated
        text_surface = self.font.render(f"Enemies Remaining: {enemies_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (10, 10))

        if self.game_over and self.enemies_defeated >= self.TOTAL_ENEMIES_TO_DEFEAT:
            win_font = pygame.font.SysFont('Consolas', 60, bold=True)
            win_text = win_font.render("VICTORY", True, (255, 215, 0))
            text_rect = win_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(win_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "enemies_defeated": self.enemies_defeated,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        # This is to ensure we capture key presses correctly
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Transpose and convert for Pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
            # Optional: auto-reset on finish
            # obs, info = env.reset()
            # total_reward = 0
            
        clock.tick(30) # Run at 30 FPS

    env.close()