
# Generated: 2025-08-28T02:03:23.749054
# Source Brief: brief_01583.md
# Brief Index: 1583

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game entities
class Particle:
    def __init__(self, pos, vel, color, radius, lifespan):
        self.pos = list(pos)
        self.vel = list(vel)
        self.color = color
        self.radius = radius
        self.lifespan = lifespan
        self.age = 0

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.age += 1
        self.radius = max(0, self.radius * 0.95)
        return self.age >= self.lifespan

    def draw(self, surface):
        alpha = int(255 * (1 - self.age / self.lifespan))
        if alpha > 0:
            pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), int(self.radius), (*self.color, alpha))
            pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), int(self.radius), (*self.color, alpha))

class Projectile:
    def __init__(self, pos, target, speed, damage):
        self.pos = np.array(pos, dtype=float)
        self.target = target
        self.speed = speed
        self.damage = damage
        self.terminated = False
        
        direction = np.array(target.pos) - self.pos
        dist = np.linalg.norm(direction)
        if dist > 0:
            self.vel = (direction / dist) * speed
        else:
            self.vel = np.array([0, -speed])

    def update(self):
        self.pos += self.vel
        if np.linalg.norm(self.pos - self.target.pos) < self.target.radius:
            self.target.take_damage(self.damage)
            self.terminated = True
            return 0.1 # Reward for hitting
        return 0

    def draw(self, surface):
        end_pos = self.pos + self.vel * 1.5
        pygame.draw.line(surface, (255, 255, 0), self.pos, end_pos, 2)

class Enemy:
    def __init__(self, pos, health, speed, radius, color, base_pos):
        self.pos = np.array(pos, dtype=float)
        self.health = health
        self.max_health = health
        self.speed = speed
        self.radius = radius
        self.color = color
        self.base_pos = np.array(base_pos, dtype=float)
        self.is_slowed = False
        self.target_block = None

    def take_damage(self, amount):
        self.health -= amount
        return self.health <= 0

    def update(self, blocks, grid_size):
        # Reset slow status
        self.is_slowed = False
        
        # Find target direction (base)
        direction = self.base_pos - self.pos
        dist_to_base = np.linalg.norm(direction)
        
        if dist_to_base > 0:
            direction_norm = direction / dist_to_base
        else:
            direction_norm = np.array([0,0])

        current_speed = self.speed * 0.5 if self.is_slowed else self.speed
        next_pos = self.pos + direction_norm * current_speed

        # Check for blocking blocks
        next_grid_pos = (int(next_pos[0] // grid_size), int(next_pos[1] // grid_size))
        
        if self.target_block and not self.target_block.is_alive:
            self.target_block = None

        if not self.target_block and next_grid_pos in blocks and blocks[next_grid_pos].type in ["WALL", "TURRET", "MINE"]:
            self.target_block = blocks[next_grid_pos]

        if self.target_block:
            # Attack block
            if self.target_block.take_damage(1):
                # block destroyed
                self.target_block = None
        else:
            # Move
            self.pos = next_pos

    def draw(self, surface):
        # Body
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), self.radius, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), self.radius, (255, 255, 255))
        # Health bar
        if self.health < self.max_health:
            health_ratio = self.health / self.max_health
            bar_width = self.radius * 2
            bar_height = 4
            bar_x = self.pos[0] - self.radius
            bar_y = self.pos[1] - self.radius - bar_height - 2
            pygame.draw.rect(surface, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(surface, (255, 0, 0), (bar_x, bar_y, bar_width * health_ratio, bar_height))

class Block:
    BLOCK_TYPES = {
        "WALL": {"color": (0, 150, 255), "cost": 1, "health": 200},
        "SLOW": {"color": (255, 200, 0), "cost": 2, "health": 50, "range": 60},
        "TURRET": {"color": (200, 50, 255), "cost": 5, "health": 100, "range": 120, "fire_rate": 30, "damage": 10},
        "MINE": {"color": (255, 100, 0), "cost": 3, "health": 10, "range": 40, "damage": 50},
        "REPAIR": {"color": (50, 255, 50), "cost": 4, "heal": 50},
    }

    def __init__(self, grid_pos, type_name, grid_size):
        self.grid_pos = grid_pos
        self.type = type_name
        self.props = self.BLOCK_TYPES[type_name]
        self.color = self.props["color"]
        self.health = self.props.get("health", 1)
        self.max_health = self.health
        self.is_alive = True
        self.cooldown = 0
        self.pos = (grid_pos[0] * grid_size + grid_size / 2, grid_pos[1] * grid_size + grid_size / 2)

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.is_alive = False
            return True # Destroyed
        return False

    def update(self):
        if self.cooldown > 0:
            self.cooldown -= 1

    def draw(self, surface, grid_size):
        rect = pygame.Rect(self.grid_pos[0] * grid_size, self.grid_pos[1] * grid_size, grid_size, grid_size)
        
        if self.type == "SLOW":
            alpha = 60
            pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), self.props["range"], (*self.color, alpha))
            pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), self.props["range"], (*self.color, alpha+20))

        inner_rect = rect.inflate(-4, -4)
        pygame.draw.rect(surface, self.color, inner_rect, border_radius=3)
        pygame.draw.rect(surface, tuple(min(255, c + 50) for c in self.color), inner_rect, 1, border_radius=3)
        
        if self.type == "TURRET":
            center = inner_rect.center
            pygame.draw.circle(surface, (255,255,255), center, 4)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Use arrow keys to move the cursor. Press space to place a block. Hold shift to cycle block types."
    game_description = "Defend your core against waves of enemies by strategically placing defensive blocks."
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 20
    GRID_W, GRID_H = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE
    MAX_STEPS = 30000 # Approx 16 mins at 30fps
    MAX_WAVES = 20
    
    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (40, 45, 50)
    COLOR_BASE = (50, 200, 50)
    COLOR_ENEMY_A = (220, 50, 50)
    COLOR_ENEMY_B = (250, 120, 50)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 14)
        self.font_m = pygame.font.SysFont("Consolas", 18, bold=True)

        self.block_types_list = list(Block.BLOCK_TYPES.keys())
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0

        self.base_grid_pos = (self.GRID_W // 2, self.GRID_H // 2)
        self.base_pos = (self.base_grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE/2, self.base_grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE/2)
        self.max_base_health = 500
        self.base_health = self.max_base_health

        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2 - 3]
        self.selected_block_idx = 0

        self.blocks = {}
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.wave_number = 0
        self.wave_in_progress = False
        self.between_wave_timer = 90 # 3 seconds

        self.prev_space_held = False
        self.prev_shift_held = False

        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
        
        self.reward_this_step = 0
        terminated = False

        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()
            
            self.score += self.reward_this_step
            
            if self.base_health <= 0:
                self.game_over = True
                terminated = True
                self.reward_this_step -= 100 # Lose penalty
                self._create_explosion(self.base_pos, 100, (255, 255, 100))
            elif self.wave_number > self.MAX_WAVES:
                self.game_over = True
                terminated = True
                self.reward_this_step += 500 # Win bonus
            elif self.steps >= self.MAX_STEPS:
                self.game_over = True
                terminated = True
        
        self.steps += 1
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)
        
        # --- Block Placement ---
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            self._place_block()
        
        # --- Cycle Block Type ---
        shift_pressed = shift_held and not self.prev_shift_held
        if shift_pressed:
            self.selected_block_idx = (self.selected_block_idx + 1) % len(self.block_types_list)

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _place_block(self):
        grid_pos = tuple(self.cursor_pos)
        if grid_pos not in self.blocks and grid_pos != self.base_grid_pos:
            block_type_name = self.block_types_list[self.selected_block_idx]
            
            if block_type_name == "REPAIR":
                heal_amount = Block.BLOCK_TYPES["REPAIR"]["heal"]
                self.base_health = min(self.max_base_health, self.base_health + heal_amount)
                self.reward_this_step += 5 # Small reward for healing
                # Sound: Repair
                self._create_explosion(self.base_pos, 20, (50, 255, 50))
            else:
                self.blocks[grid_pos] = Block(grid_pos, block_type_name, self.GRID_SIZE)
                self.reward_this_step -= 0.01 # Cost for placing
                # Sound: Place block

    def _update_game_state(self):
        # Update wave logic
        if not self.wave_in_progress:
            self.between_wave_timer -= 1
            if self.between_wave_timer <= 0:
                self._start_next_wave()

        # Update blocks and turrets
        new_projectiles = []
        for block in list(self.blocks.values()):
            block.update()
            if not block.is_alive:
                del self.blocks[block.grid_pos]
                continue
            
            if block.type == "TURRET" and block.cooldown == 0:
                target = self._find_closest_enemy(block.pos, block.props["range"])
                if target:
                    new_projectiles.append(Projectile(block.pos, target, 5, block.props["damage"]))
                    block.cooldown = block.props["fire_rate"]
                    # Sound: Turret fire
            
            if block.type == "MINE":
                target = self._find_closest_enemy(block.pos, block.props["range"])
                if target:
                    self._detonate_mine(block)
                    block.is_alive = False # Mine is consumed
        self.projectiles.extend(new_projectiles)

        # Update and move enemies
        for enemy in self.enemies:
            # Check for slow fields
            for block in self.blocks.values():
                if block.type == "SLOW" and np.linalg.norm(enemy.pos - block.pos) < block.props["range"]:
                    enemy.is_slowed = True
                    break

            enemy.update(self.blocks, self.GRID_SIZE)
            
            # Check collision with base
            if np.linalg.norm(enemy.pos - self.base_pos) < self.GRID_SIZE:
                self.base_health -= 10
                self.reward_this_step -= 10
                enemy.health = 0 # Enemy is destroyed on impact
                self._create_explosion(self.base_pos, 30, (255, 100, 0))
                # Sound: Base hit

        # Update projectiles
        for p in self.projectiles:
            hit_reward = p.update()
            if hit_reward > 0:
                self.reward_this_step += hit_reward
                self._create_explosion(p.pos, 5, (255, 255, 200))

        # Cleanup dead/terminated entities
        dead_enemies = [e for e in self.enemies if e.health <= 0]
        for _ in dead_enemies:
            self.reward_this_step += 1 # Reward for destroying enemy
        
        if dead_enemies:
            for enemy in dead_enemies:
                self._create_explosion(enemy.pos, 20, enemy.color)
                # Sound: Enemy explosion
        
        self.enemies = [e for e in self.enemies if e.health > 0]
        self.projectiles = [p for p in self.projectiles if not p.terminated and 0 < p.pos[0] < self.WIDTH and 0 < p.pos[1] < self.HEIGHT]
        self.particles = [p for p in self.particles if not p.update()]

        # Check for wave completion
        if self.wave_in_progress and not self.enemies:
            self.wave_in_progress = False
            self.between_wave_timer = 150 # 5 seconds
            self.reward_this_step += 100 # Wave complete bonus

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            return
            
        self.wave_in_progress = True
        num_enemies = 2 + self.wave_number
        
        enemy_speed = 0.5 + 0.05 * (self.wave_number // 5)
        enemy_health = 50 + 10 * self.wave_number
        enemy_color = self.COLOR_ENEMY_A
        if self.wave_number > 5: enemy_color = self.COLOR_ENEMY_B
        
        for _ in range(num_enemies):
            side = self.np_random.integers(4)
            if side == 0: x, y = self.np_random.uniform(0, self.WIDTH), -20
            elif side == 1: x, y = self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT)
            elif side == 2: x, y = self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20
            else: x, y = -20, self.np_random.uniform(0, self.HEIGHT)
            
            self.enemies.append(Enemy((x, y), enemy_health, enemy_speed, 8, enemy_color, self.base_pos))

    def _find_closest_enemy(self, pos, max_range):
        closest_enemy = None
        min_dist_sq = max_range ** 2
        for enemy in self.enemies:
            dist_sq = (enemy.pos[0] - pos[0])**2 + (enemy.pos[1] - pos[1])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_enemy = enemy
        return closest_enemy
    
    def _detonate_mine(self, mine_block):
        # Sound: Mine explosion
        self._create_explosion(mine_block.pos, mine_block.props["range"]*1.5, mine_block.color)
        for enemy in self.enemies:
            if np.linalg.norm(enemy.pos - mine_block.pos) < mine_block.props["range"]:
                if enemy.take_damage(mine_block.props["damage"]):
                    self.reward_this_step += 1 # Kill credit
                else:
                    self.reward_this_step += 0.1 # Damage credit

    def _create_explosion(self, pos, max_radius, color):
        num_particles = int(max_radius/2)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            radius = self.np_random.uniform(max_radius/4, max_radius/2)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append(Particle(pos, vel, color, radius, lifespan))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw blocks
        for block in self.blocks.values():
            block.draw(self.screen, self.GRID_SIZE)

        # Draw base
        base_rect = pygame.Rect(self.base_grid_pos[0] * self.GRID_SIZE, self.base_grid_pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect.inflate(-4,-4), border_radius=3)
        pygame.draw.rect(self.screen, (255,255,255), base_rect.inflate(-4,-4), 1, border_radius=3)

        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(self.screen)

        # Draw projectiles
        for p in self.projectiles:
            p.draw(self.screen)
        
        # Draw particles
        for particle in self.particles:
            particle.draw(self.screen)

        # Draw cursor
        if not self.game_over:
            cursor_rect = pygame.Rect(self.cursor_pos[0] * self.GRID_SIZE, self.cursor_pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            selected_type = self.block_types_list[self.selected_block_idx]
            color = Block.BLOCK_TYPES[selected_type]["color"]
            
            # Ghost block
            s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(s, (*color, 100), (2, 2, self.GRID_SIZE-4, self.GRID_SIZE-4), border_radius=3)
            self.screen.blit(s, cursor_rect.topleft)
            
            # Cursor border
            pygame.draw.rect(self.screen, (255, 255, 255), cursor_rect, 1)

    def _render_ui(self):
        # Wave info
        wave_text = f"WAVE: {self.wave_number}/{self.MAX_WAVES}"
        if not self.wave_in_progress and self.wave_number <= self.MAX_WAVES:
            wave_text += f" (Next in {self.between_wave_timer/30:.1f}s)"
        
        text_surf = self.font_m.render(wave_text, True, (255, 255, 255))
        self.screen.blit(text_surf, (10, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        text_surf = self.font_m.render(score_text, True, (255, 255, 255))
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))

        # Base Health Bar
        health_ratio = self.base_health / self.max_base_health
        bar_width = 100
        bar_height = 10
        bar_x = self.base_pos[0] - bar_width/2
        bar_y = self.base_pos[1] + self.GRID_SIZE/2 + 5
        pygame.draw.rect(self.screen, (50,0,0), (bar_x, bar_y, bar_width, bar_height), border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_BASE, (bar_x, bar_y, bar_width * health_ratio, bar_height), border_radius=2)

        # Selected Block UI
        selected_type_name = self.block_types_list[self.selected_block_idx]
        props = Block.BLOCK_TYPES[selected_type_name]
        color = props["color"]
        
        ui_box = pygame.Rect(10, self.HEIGHT - 50, 150, 40)
        pygame.draw.rect(self.screen, (10,10,10,200), ui_box, border_radius=5)
        
        pygame.draw.rect(self.screen, color, (18, self.HEIGHT - 42, 24, 24), border_radius=3)
        
        type_text = self.font_m.render(selected_type_name, True, (255, 255, 255))
        self.screen.blit(type_text, (50, self.HEIGHT - 44))

        # Game Over Text
        if self.game_over:
            outcome_text = "VICTORY!" if self.wave_number > self.MAX_WAVES else "GAME OVER"
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            font_l = pygame.font.SysFont("Consolas", 48, bold=True)
            text_surf = font_l.render(outcome_text, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert self.base_health == self.max_base_health
        assert self.wave_number == 1
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Requires pygame to be installed and not in a headless environment
    try:
        import os
        # os.environ["SDL_VIDEODRIVER"] = "dummy" # For headless
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Block Fortress Defense")
        clock = pygame.time.Clock()

        running = True
        total_reward = 0
        
        while running:
            movement, space, shift = 0, 0, 0
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # The observation is the rendered screen, so we just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Waves Survived: {info['wave']-1}")
                # Wait a bit before resetting
                pygame.time.wait(3000)
                obs, info = env.reset()
                total_reward = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            clock.tick(30)
            
        env.close()

    except Exception as e:
        print("\nCould not run interactive game mode.")
        print("This is expected in a headless environment.")
        print(f"Error: {e}")
        # Test if the class can be instantiated without error
        env = GameEnv()
        print("\nGym environment created successfully in headless mode.")