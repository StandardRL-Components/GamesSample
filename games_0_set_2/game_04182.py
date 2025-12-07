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


# Helper class for managing particles (visual effects)
class Particle:
    def __init__(self, x, y, color, lifetime, velocity, text=None, font=None):
        self.x = x
        self.y = y
        self.color = color
        self.lifetime = lifetime
        self.initial_lifetime = lifetime
        self.velocity = velocity
        self.text = text
        self.font = font

    def update(self):
        self.x += self.velocity[0]
        self.y += self.velocity[1]
        self.lifetime -= 1

    def draw(self, surface):
        alpha = int(255 * (self.lifetime / self.initial_lifetime))
        alpha = max(0, min(255, alpha))
        
        if self.text and self.font:
            text_surf = self.font.render(self.text, True, self.color)
            text_surf.set_alpha(alpha)
            surface.blit(text_surf, (int(self.x), int(self.y)))
        else:
            radius = int(3 * (self.lifetime / self.initial_lifetime))
            if radius > 0:
                pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), radius, (*self.color, alpha))

# Helper class for enemies
class Enemy:
    def __init__(self, x, y, enemy_type, difficulty_modifier):
        self.x = x
        self.y = y
        self.type = enemy_type
        self.difficulty_modifier = difficulty_modifier
        
        if enemy_type == 'blue':
            self.base_health = 20
            self.base_damage = 5
            self.color = (100, 150, 255)
        elif enemy_type == 'green':
            self.base_health = 30
            self.base_damage = 10
            self.color = (100, 255, 150)
        else: # purple
            self.base_health = 10
            self.base_damage = 15
            self.color = (200, 100, 255)
            
        self.max_health = int(self.base_health * self.difficulty_modifier)
        self.health = self.max_health
        self.damage = int(self.base_damage * self.difficulty_modifier)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Space to attack in your last moved direction. Collect gold and reach the chest!"
    )

    game_description = (
        "Explore a dangerous dungeon, defeat monsters, and collect treasure. Reach the final chest with enough gold to win."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.ui_font = pygame.font.Font(None, 28)
        self.particle_font = pygame.font.Font(None, 22)
        
        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_FLOOR = (50, 50, 70)
        self.COLOR_WALL = (100, 100, 120)
        self.COLOR_WALL_LIGHT = (130, 130, 150)
        self.COLOR_WALL_DARK = (70, 70, 90)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_PLAYER_OUTLINE = (255, 150, 150)
        self.COLOR_GOLD = (255, 223, 0)
        self.COLOR_CHEST = (200, 150, 50)
        
        # Game constants
        self.GRID_WIDTH = 20
        self.GRID_HEIGHT = 12
        self.TILE_SIZE = 32
        self.MAX_STEPS = 1000
        self.WIN_GOLD_REQ = 50
        self.TOTAL_ROOMS = 10

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_max_health = None
        self.player_gold = None
        self.player_facing = None
        self.enemies = []
        self.gold_coins = []
        self.particles = []
        self.current_room_data = None
        self.dungeon_layout = []
        self.current_room_index = 0
        self.rooms_cleared = 0
        self.difficulty_modifier = 1.0
        self.chest_pos = None

        self.steps = 0
        self.game_over = False
        
        # self.reset() is called by gym.make, no need to call it here.
    
    def _generate_dungeon(self):
        self.dungeon_layout = []
        templates = [
            # Template 0: Open room
            ["WWWWWWWWWWWWWWWWWWWW",
             "W..................W",
             "E..................E",
             "W..................W",
             "W..................W",
             "W..................W",
             "W..................W",
             "W..................W",
             "W..................W",
             "E..................E",
             "W..................W",
             "WWWWWWWWWWWWWWWWWWWW"],
            # Template 1: Pillars
            ["WWWWWWWWWWWWWWWWWWWW",
             "W..................W",
             "E....WW......WW....E",
             "W....WW......WW....W",
             "W..................W",
             "W..................W",
             "W..................W",
             "W....WW......WW....W",
             "W....WW......WW....W",
             "E..................E",
             "W..................W",
             "WWWWWWWWWWWWWWWWWWWW"],
            # Template 2: Center block
            ["WWWWWWWWWWWWWWWWWWWW",
             "W..................W",
             "E..................E",
             "W..................W",
             "W......WWWWW.......W",
             "W......WWWWW.......W",
             "W......WWWWW.......W",
             "W......WWWWW.......W",
             "W..................W",
             "E..................E",
             "W..................W",
             "WWWWWWWWWWWWWWWWWWWW"],
        ]
        for _ in range(self.TOTAL_ROOMS):
            self.dungeon_layout.append(random.choice(templates))

    def _load_room(self, room_index):
        self.current_room_index = room_index
        self.current_room_data = np.array([list(row) for row in self.dungeon_layout[room_index]])
        
        # Clear entities
        self.enemies.clear()
        self.gold_coins.clear()
        self.chest_pos = None
        
        # Place player
        entrances = np.argwhere(self.current_room_data == 'E')
        if entrances.size > 0:
            entry_pos = random.choice(entrances)
            self.player_pos = [entry_pos[1], entry_pos[0]]
        else:
            self.player_pos = [1, 1]

        # Place entities
        floor_tiles = np.argwhere(self.current_room_data == '.')
        random.shuffle(floor_tiles)
        
        num_enemies = min(len(floor_tiles) -1, random.randint(2, 4) + self.rooms_cleared // 2)
        for i in range(num_enemies):
            pos = floor_tiles[i]
            enemy_type = random.choice(['blue', 'green', 'purple'])
            self.enemies.append(Enemy(pos[1], pos[0], enemy_type, self.difficulty_modifier))

        num_gold = min(len(floor_tiles) - num_enemies -1, random.randint(3, 6))
        for i in range(num_enemies, num_enemies + num_gold):
            pos = floor_tiles[i]
            self.gold_coins.append([pos[1], pos[0]])

        # Place chest in the last room
        if self.current_room_index == self.TOTAL_ROOMS - 1 and len(floor_tiles) > num_enemies + num_gold:
            pos = floor_tiles[-1]
            self.chest_pos = [pos[1], pos[0]]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        
        self.steps = 0
        self.game_over = False
        self.player_max_health = 100
        self.player_health = self.player_max_health
        self.player_gold = 0
        self.player_facing = [1, 0] # Start facing right
        self.particles.clear()
        
        self.rooms_cleared = 0
        self.difficulty_modifier = 1.0
        self._generate_dungeon()
        self._load_room(0)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        player_acted = False

        # --- Player Turn ---
        # Attack takes precedence over movement
        if space_pressed:
            player_acted = True
            # Sound: Sword slash
            attack_pos = [self.player_pos[0] + self.player_facing[0], self.player_pos[1] + self.player_facing[1]]
            self._create_slash_effect(attack_pos)
            
            for enemy in self.enemies:
                if [enemy.x, enemy.y] == attack_pos:
                    enemy.health -= 20
                    self._create_damage_particle(enemy.x, enemy.y, "20")
                    if enemy.health <= 0:
                        reward += 1.0 # Defeat enemy reward
                        self._create_death_effect(enemy.x, enemy.y, enemy.color)
                        # Sound: Enemy defeat
                    break # Can only hit one enemy
            self.enemies = [e for e in self.enemies if e.health > 0]

        elif movement > 0:
            player_acted = True
            move_map = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]} # Up, Down, Left, Right
            move_dir = move_map.get(movement, [0, 0])
            self.player_facing = move_dir
            
            new_pos = [self.player_pos[0] + move_dir[0], self.player_pos[1] + move_dir[1]]

            # Boundary check to prevent IndexError
            if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
                # Check for room transition
                if self.current_room_data[new_pos[1], new_pos[0]] == 'E':
                    self.rooms_cleared += 1
                    if self.rooms_cleared > 0 and self.rooms_cleared % 5 == 0:
                        self.difficulty_modifier *= 1.05
                    self._load_room(min(self.current_room_index + 1, self.TOTAL_ROOMS - 1))
                    # Sound: Door open
                # Check for wall collision
                elif self.current_room_data[new_pos[1], new_pos[0]] != 'W':
                    self.player_pos = new_pos
                    # Sound: Footstep
                    
                    # Check for gold collection
                    collected_coin = -1
                    for i, coin_pos in enumerate(self.gold_coins):
                        if coin_pos == self.player_pos:
                            self.player_gold += 10
                            reward += 0.1 # Gold collection reward
                            collected_coin = i
                            self._create_gold_particle(coin_pos[0], coin_pos[1], "+10")
                            # Sound: Coin collect
                            break
                    if collected_coin != -1:
                        self.gold_coins.pop(collected_coin)
            # If move is out of bounds, action is consumed but nothing happens.
        
        # --- Enemy Turn ---
        if player_acted:
            for enemy in self.enemies:
                dist_x = self.player_pos[0] - enemy.x
                dist_y = self.player_pos[1] - enemy.y

                # Attack if adjacent
                if abs(dist_x) + abs(dist_y) == 1:
                    self.player_health -= enemy.damage
                    self._create_damage_particle(self.player_pos[0], self.player_pos[1], f"-{enemy.damage}", self.COLOR_PLAYER)
                    # Sound: Player hurt
                # Move towards player
                else:
                    move_x, move_y = 0, 0
                    if abs(dist_x) > abs(dist_y):
                        move_x = np.sign(dist_x)
                    else:
                        move_y = np.sign(dist_y)
                    
                    new_enemy_pos = [enemy.x + move_x, enemy.y + move_y]
                    if self.current_room_data[new_enemy_pos[1], new_enemy_pos[0]] != 'W':
                        is_occupied = False
                        for other_enemy in self.enemies:
                            if [other_enemy.x, other_enemy.y] == new_enemy_pos:
                                is_occupied = True
                                break
                        if not is_occupied and new_enemy_pos != self.player_pos:
                            enemy.x, enemy.y = new_enemy_pos[0], new_enemy_pos[1]

        self._update_particles()
        self.steps += 1
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated:
            self.game_over = True
            if self.player_health <= 0:
                reward = -100.0 # Death penalty
            elif self.chest_pos and self.player_pos == self.chest_pos and self.player_gold >= self.WIN_GOLD_REQ:
                reward = 100.0 # Victory reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _check_termination(self):
        if self.player_health <= 0:
            return True
        if self.chest_pos and self.player_pos == self.chest_pos:
            if self.player_gold >= self.WIN_GOLD_REQ:
                # Victory
                return True
        return False

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

    def _create_particle_effect(self, x, y, color, count, text=None, font=None):
        px, py = (x + 0.5) * self.TILE_SIZE, (y + 0.5) * self.TILE_SIZE
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(15, 30)
            self.particles.append(Particle(px, py, color, lifetime, vel))
            
    def _create_death_effect(self, x, y, color):
        self._create_particle_effect(x, y, color, 20)

    def _create_damage_particle(self, x, y, text, color=(255,255,255)):
        px, py = (x + 0.5) * self.TILE_SIZE - 10, y * self.TILE_SIZE
        vel = [0, -1]
        lifetime = 30
        self.particles.append(Particle(px, py, color, lifetime, vel, text, self.particle_font))
        
    def _create_gold_particle(self, x, y, text):
        self._create_damage_particle(x, y, text, self.COLOR_GOLD)
        
    def _create_slash_effect(self, pos):
        px, py = (pos[0] + 0.5) * self.TILE_SIZE, (pos[1] + 0.5) * self.TILE_SIZE
        color = (200, 200, 200)
        for i in range(5):
            vel = [self.player_facing[0] * i*0.5, self.player_facing[1] * i*0.5]
            self.particles.append(Particle(px, py, color, 5, vel))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        ox, oy = (self.screen_width - self.GRID_WIDTH * self.TILE_SIZE) // 2, \
                 (self.screen_height - self.GRID_HEIGHT * self.TILE_SIZE) // 2

        # Draw grid and walls
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(ox + x * self.TILE_SIZE, oy + y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                char = self.current_room_data[y, x]
                if char == 'W':
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                    pygame.draw.line(self.screen, self.COLOR_WALL_DARK, rect.bottomleft, rect.bottomright, 2)
                    pygame.draw.line(self.screen, self.COLOR_WALL_DARK, rect.topright, rect.bottomright, 2)
                    pygame.draw.line(self.screen, self.COLOR_WALL_LIGHT, rect.topleft, rect.bottomleft, 2)
                    pygame.draw.line(self.screen, self.COLOR_WALL_LIGHT, rect.topleft, rect.topright, 2)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)
                    if char == 'E':
                         pygame.draw.rect(self.screen, (60,60,80), rect, 3)

        # Draw chest
        if self.chest_pos:
            rect = pygame.Rect(ox + self.chest_pos[0] * self.TILE_SIZE, oy + self.chest_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CHEST, rect.inflate(-4, -4))
            pygame.draw.rect(self.screen, self.COLOR_GOLD, rect.inflate(-12, -20))

        # Draw gold
        for gx, gy in self.gold_coins:
            pos_x = int(ox + (gx + 0.5) * self.TILE_SIZE)
            pos_y = int(oy + (gy + 0.5) * self.TILE_SIZE)
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, self.TILE_SIZE // 4, self.COLOR_GOLD)
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, self.TILE_SIZE // 4, self.COLOR_GOLD)

        # Draw enemies
        for enemy in self.enemies:
            rect = pygame.Rect(ox + enemy.x * self.TILE_SIZE, oy + enemy.y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, enemy.color, rect.inflate(-4, -4))
            # Health bar for enemy
            if enemy.health < enemy.max_health:
                hp_ratio = enemy.health / enemy.max_health
                hp_bar_rect = pygame.Rect(rect.left, rect.top - 7, rect.width, 5)
                pygame.draw.rect(self.screen, (255,0,0), hp_bar_rect)
                pygame.draw.rect(self.screen, (0,255,0), (hp_bar_rect.left, hp_bar_rect.top, hp_bar_rect.width * hp_ratio, hp_bar_rect.height))

        # Draw player
        bob = math.sin(self.steps * 0.4) * 2 if not self.game_over else 0
        player_rect = pygame.Rect(ox + self.player_pos[0] * self.TILE_SIZE, oy + self.player_pos[1] * self.TILE_SIZE + bob, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect.inflate(-2, -2))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-6, -6))

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Health Bar
        hp_ratio = max(0, self.player_health / self.player_max_health)
        hp_bar_bg = pygame.Rect(10, 10, 200, 20)
        hp_bar_fg = pygame.Rect(10, 10, 200 * hp_ratio, 20)
        pygame.draw.rect(self.screen, (100, 0, 0), hp_bar_bg)
        pygame.draw.rect(self.screen, (0, 200, 0), hp_bar_fg)
        pygame.draw.rect(self.screen, (255, 255, 255), hp_bar_bg, 2)
        hp_text = self.ui_font.render(f"HP: {self.player_health}/{self.player_max_health}", True, (255, 255, 255))
        self.screen.blit(hp_text, (15, 11))

        # Gold Counter
        gold_text = self.ui_font.render(f"GOLD: {self.player_gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (self.screen_width - gold_text.get_width() - 10, 10))

        # Room Counter
        room_text = self.ui_font.render(f"ROOM: {self.current_room_index + 1}/{self.TOTAL_ROOMS}", True, (200, 200, 200))
        self.screen.blit(room_text, (self.screen_width - room_text.get_width() - 10, 40))

        # Game Over/Win Text
        if self.game_over:
            if self.player_health <= 0:
                msg = "GAME OVER"
                color = self.COLOR_PLAYER
            elif self.chest_pos and self.player_pos == self.chest_pos and self.player_gold >= self.WIN_GOLD_REQ:
                msg = "VICTORY!"
                color = self.COLOR_GOLD
            else: # Time out
                msg = "TIME'S UP"
                color = (200, 200, 200)

            end_text = pygame.font.Font(None, 72).render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.screen_width/2, self.screen_height/2))
            pygame.draw.rect(self.screen, self.COLOR_BG, text_rect.inflate(20,20))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "gold": self.player_gold,
            "health": self.player_health,
            "steps": self.steps,
            "room": self.current_room_index + 1,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It requires the os.environ line at the top to be commented out
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    running = True
    terminated = False
    truncated = False
    
    # Pygame setup for manual play
    pygame.display.set_caption("Dungeon Crawler")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    clock = pygame.time.Clock()

    action = np.array([0, 0, 0])

    while running:
        # Event handling for manual play
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset(seed=42)
                    terminated = False
                    truncated = False
                    continue
                
                # Map keys to MultiDiscrete action space for a single turn
                movement = 0 # 0=none
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                
                space_pressed = 1 if event.key == pygame.K_SPACE else 0
                
                if movement > 0 or space_pressed > 0:
                    action = np.array([movement, space_pressed, 0]) # Shift not used in manual play
                    action_taken = True

        if (terminated or truncated):
            # Just render the final state, wait for 'r' to reset
            pass
        elif action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Gold: {info['gold']}, Health: {info['health']}, Reward: {reward:.2f}, Term: {terminated}, Trunc: {truncated}")
            if terminated or truncated:
                print("--- Episode Finished ---")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60)

    env.close()