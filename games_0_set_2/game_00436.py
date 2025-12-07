
# Generated: 2025-08-27T13:38:37.269945
# Source Brief: brief_00436.md
# Brief Index: 436

        
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


# --- Helper Classes for Game Entities ---

class Particle:
    """A simple class for a particle effect."""
    def __init__(self, x, y, color, life, size_range=(2, 5), velocity_range=(-2, 2)):
        self.x = x
        self.y = y
        self.vx = random.uniform(velocity_range[0], velocity_range[1])
        self.vy = random.uniform(velocity_range[0], velocity_range[1])
        self.color = color
        self.life = life
        self.initial_life = life
        self.size = random.uniform(size_range[0], size_range[1])

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.initial_life))
            temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(
                temp_surf, int(self.size), int(self.size), int(self.size), (*self.color, alpha)
            )
            surface.blit(temp_surf, (int(self.x - self.size), int(self.y - self.size)))

class FloatingText:
    """A class for displaying floating text like damage numbers."""
    def __init__(self, x, y, text, font, color, life=45):
        self.x = x
        self.y = y
        self.text = text
        self.font = font
        self.color = color
        self.life = life
        self.initial_life = life
        self.vy = -0.75

    def update(self):
        self.y += self.vy
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = max(0, int(255 * (self.life / self.initial_life)))
            text_surface = self.font.render(self.text, True, self.color)
            text_surface.set_alpha(alpha)
            surface.blit(text_surface, (int(self.x), int(self.y)))

class Monster:
    """A class for monsters in the dungeon."""
    BASE_STATS = {
        "slime": {"health": 20, "damage": 5, "color": (100, 220, 120), "size": 12},
        "goblin": {"health": 30, "damage": 10, "color": (220, 100, 80), "size": 14},
        "orc": {"health": 40, "damage": 15, "color": (240, 60, 60), "size": 18},
    }

    def __init__(self, x, y, m_type, level_modifier, np_random):
        self.x = x
        self.y = y
        self.type = m_type
        self.np_random = np_random
        
        stats = self.BASE_STATS[m_type]
        self.max_health = int(stats["health"] * level_modifier)
        self.health = self.max_health
        self.damage = int(stats["damage"] * level_modifier)
        self.color = stats["color"]
        self.size = stats["size"]
        self.bob_offset = 0
        self.bob_speed = random.uniform(0.1, 0.2)

    def update_animation(self):
        self.bob_offset = math.sin(pygame.time.get_ticks() * self.bob_speed) * 2

    def take_damage(self, amount):
        self.health = max(0, self.health - amount)
        return self.health <= 0

    def draw(self, surface):
        pos_x, pos_y = int(self.x), int(self.y + self.bob_offset)
        
        # Health bar
        if self.health < self.max_health:
            bar_width = self.size * 2
            bar_height = 4
            health_ratio = self.health / self.max_health
            pygame.draw.rect(surface, (50, 0, 0), (pos_x - self.size, pos_y - self.size - 8, bar_width, bar_height))
            pygame.draw.rect(surface, (255, 0, 0), (pos_x - self.size, pos_y - self.size - 8, bar_width * health_ratio, bar_height))
        
        # Body
        pygame.draw.circle(surface, self.color, (pos_x, pos_y), self.size)
        pygame.draw.circle(surface, (0, 0, 0), (pos_x, pos_y), self.size, 2)
        
        # Eyes
        eye_x_offset = self.size * 0.3
        eye_y_offset = self.size * 0.2
        pygame.draw.circle(surface, (255, 255, 255), (int(pos_x - eye_x_offset), int(pos_y - eye_y_offset)), 3)
        pygame.draw.circle(surface, (255, 255, 255), (int(pos_x + eye_x_offset), int(pos_y - eye_y_offset)), 3)
        pygame.draw.circle(surface, (0,0,0), (int(pos_x - eye_x_offset), int(pos_y - eye_y_offset)), 1)
        pygame.draw.circle(surface, (0,0,0), (int(pos_x + eye_x_offset), int(pos_y - eye_y_offset)), 1)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move, Space to attack. Do nothing to defend. "
        "One action per turn."
    )

    game_description = (
        "Explore a dungeon, fight monsters, and collect gold. Reach the 5th room to win."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # --- Visuals & Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_WALL = (60, 70, 80)
        self.COLOR_DOOR = (100, 80, 50)
        self.COLOR_PLAYER = (50, 200, 50)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_SHIELD = (100, 150, 255, 100)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_DMG_PLAYER = (255, 80, 80)
        self.COLOR_DMG_ENEMY = (255, 255, 255)
        self.COLOR_GOLD_TEXT = (255, 220, 100)
        
        self.FONT_UI = pygame.font.SysFont("monospace", 18, bold=True)
        self.FONT_COMBAT = pygame.font.SysFont("monospace", 16, bold=True)
        self.FONT_GAME_OVER = pygame.font.SysFont("monospace", 48, bold=True)
        
        # --- Game Constants ---
        self.MAX_STEPS = 1000
        self.WIN_ROOM = 5
        self.PLAYER_SIZE = 12
        self.PLAYER_ATTACK_RANGE = 50
        self.PLAYER_ATTACK_COOLDOWN = 10
        
        self.WALL_THICKNESS = 20
        self.ROOM_BOUNDS = (
            self.WALL_THICKNESS, self.WALL_THICKNESS,
            self.width - self.WALL_THICKNESS * 2, self.height - self.WALL_THICKNESS * 2
        )
        
        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_max_health = 100
        self.player_gold = None
        self.room_number = None
        self.monsters = []
        self.gold_piles = []
        self.particles = []
        self.floating_texts = []
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = np.array([self.width / 2, self.height - 60], dtype=float)
        self.player_health = self.player_max_health
        self.player_gold = 0
        self.room_number = 1
        
        self.monsters.clear()
        self.gold_piles.clear()
        self.particles.clear()
        self.floating_texts.clear()
        
        self._generate_room()
        
        return self._get_observation(), self._get_info()

    def _generate_room(self):
        self.monsters.clear()
        self.gold_piles.clear()
        
        level_modifier = 1 + (self.room_number - 1) * 0.1
        
        # Generate Monsters
        num_monsters = self.np_random.integers(
            low=min(self.room_number, 3), 
            high=min(self.room_number + 1, 4)
        )
        monster_types = ["slime", "goblin", "orc"]
        
        for _ in range(num_monsters):
            m_type = self.np_random.choice(monster_types[:min(len(monster_types), self.room_number)])
            
            x = self.np_random.uniform(self.ROOM_BOUNDS[0] + 30, self.width - self.ROOM_BOUNDS[0] - 30)
            y = self.np_random.uniform(self.ROOM_BOUNDS[1] + 30, self.height - self.ROOM_BOUNDS[1] - 80)
            
            self.monsters.append(Monster(x, y, m_type, level_modifier, self.np_random))

        # Generate Gold
        num_gold = self.np_random.integers(low=1, high=4)
        for _ in range(num_gold):
            x = self.np_random.uniform(self.ROOM_BOUNDS[0] + 20, self.width - self.ROOM_BOUNDS[0] - 20)
            y = self.np_random.uniform(self.ROOM_BOUNDS[1] + 20, self.height - self.ROOM_BOUNDS[1] - 20)
            amount = self.np_random.integers(low=10, high=31)
            self.gold_piles.append({"pos": np.array([x, y]), "amount": amount, "size": 8 + amount/5})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0
        player_is_defending = False

        # --- 1. Parse Player Action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Priority: Attack > Move > Defend
        if space_held: # ATTACK
            # Find closest monster
            target = None
            min_dist = float('inf')
            if self.monsters:
                for m in self.monsters:
                    dist = np.linalg.norm(self.player_pos - np.array([m.x, m.y]))
                    if dist < min_dist:
                        min_dist = dist
                        target = m
            
            if target and min_dist < self.PLAYER_ATTACK_RANGE:
                # // Sound: Player attack
                damage = self.np_random.integers(10, 21)
                is_dead = target.take_damage(damage)
                self.floating_texts.append(FloatingText(target.x, target.y - 20, str(damage), self.FONT_COMBAT, self.COLOR_DMG_ENEMY))
                for _ in range(10): self.particles.append(Particle(target.x, target.y, (255,255,255), 15))

                if is_dead:
                    reward += 10
                    # // Sound: Monster defeat
                    for _ in range(20): self.particles.append(Particle(target.x, target.y, target.color, 25))
                    self.monsters.remove(target)
            
        elif movement > 0: # MOVE
            move_vec = np.array([0, 0], dtype=float)
            if movement == 1: move_vec[1] = -1 # Up
            elif movement == 2: move_vec[1] = 1  # Down
            elif movement == 3: move_vec[0] = -1 # Left
            elif movement == 4: move_vec[0] = 1  # Right
            
            new_pos = self.player_pos + move_vec * 15
            
            # Wall collision
            new_pos[0] = np.clip(new_pos[0], self.ROOM_BOUNDS[0] + self.PLAYER_SIZE, self.width - self.ROOM_BOUNDS[0] - self.PLAYER_SIZE)
            new_pos[1] = np.clip(new_pos[1], self.ROOM_BOUNDS[1] + self.PLAYER_SIZE, self.height - self.ROOM_BOUNDS[1] - self.PLAYER_SIZE)
            self.player_pos = new_pos

        else: # DEFEND
            player_is_defending = True

        # --- 2. Post-Move Interactions (Gold, Door) ---
        # Gold collection
        for gold in self.gold_piles[:]:
            if np.linalg.norm(self.player_pos - gold["pos"]) < self.PLAYER_SIZE + gold["size"]:
                # // Sound: Collect gold
                self.player_gold += gold["amount"]
                self.score += gold["amount"]
                reward += gold["amount"]
                self.floating_texts.append(FloatingText(gold["pos"][0], gold["pos"][1], f"+{gold['amount']}", self.FONT_COMBAT, self.COLOR_GOLD_TEXT))
                self.gold_piles.remove(gold)
        
        # Door interaction (only if all monsters are defeated)
        door_rect = pygame.Rect(self.width/2 - 25, 0, 50, self.WALL_THICKNESS)
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE, self.player_pos[1] - self.PLAYER_SIZE, self.PLAYER_SIZE*2, self.PLAYER_SIZE*2)
        if not self.monsters and player_rect.colliderect(door_rect):
            self.room_number += 1
            if self.room_number > self.WIN_ROOM:
                self.win = True
                self.game_over = True
                reward += 100
            else:
                # // Sound: Next level
                self.player_pos = np.array([self.width / 2, self.height - 60], dtype=float)
                self._generate_room()

        # --- 3. Monsters' Turn ---
        if not self.game_over:
            for m in self.monsters:
                dir_to_player = self.player_pos - np.array([m.x, m.y])
                dist = np.linalg.norm(dir_to_player)
                
                if dist > 0:
                    dir_to_player /= dist

                if dist < m.size + self.PLAYER_SIZE + 5: # Attack range
                    # // Sound: Player takes damage
                    damage = m.damage
                    if player_is_defending:
                        damage = int(damage * 0.5)
                    
                    self.player_health -= damage
                    reward -= damage * 0.1
                    self.floating_texts.append(FloatingText(self.player_pos[0], self.player_pos[1] - 20, str(damage), self.FONT_COMBAT, self.COLOR_DMG_PLAYER))
                    for _ in range(damage): self.particles.append(Particle(self.player_pos[0], self.player_pos[1], self.COLOR_DMG_PLAYER, 20))

                elif dist > m.size + self.PLAYER_SIZE: # Move towards player
                    m.x += dir_to_player[0] * 3
                    m.y += dir_to_player[1] * 3

        # --- 4. Update Effects ---
        for p in self.particles[:]:
            p.update()
            if p.life <= 0: self.particles.remove(p)
        for t in self.floating_texts[:]:
            t.update()
            if t.life <= 0: self.floating_texts.remove(t)
        for m in self.monsters:
            m.update_animation()

        # --- 5. Check Termination ---
        terminated = False
        if self.player_health <= 0:
            self.player_health = 0
            self.game_over = True
            terminated = True
        
        if self.win:
            terminated = True

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.width, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, self.height - self.WALL_THICKNESS, self.width, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.height))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.width - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.height))

        # Door (visible if monsters are cleared)
        if not self.monsters:
            pygame.draw.rect(self.screen, self.COLOR_DOOR, (self.width/2 - 25, 0, 50, self.WALL_THICKNESS))
        
        # Gold
        for gold in self.gold_piles:
            pos = (int(gold["pos"][0]), int(gold["pos"][1]))
            size = int(gold["size"])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_GOLD)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, (255,255,255,50))

        # Monsters
        for m in self.monsters:
            m.draw(self.screen)
            
        # Player
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (px, py), self.PLAYER_SIZE)
        pygame.draw.circle(self.screen, (0,0,0), (px, py), self.PLAYER_SIZE, 2)
        # Eye to show direction (simple)
        pygame.draw.circle(self.screen, (255,255,255), (px, py - 3), 3)
        pygame.draw.circle(self.screen, (0,0,0), (px, py-3), 1)

        # Effects
        for p in self.particles: p.draw(self.screen)
        for t in self.floating_texts: t.draw(self.screen)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.player_max_health
        bar_width = 150
        pygame.draw.rect(self.screen, (50, 0, 0), (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, (200, 0, 0), (10, 10, bar_width * health_ratio, 20))
        health_text = self.FONT_UI.render(f"HP: {self.player_health}/{self.player_max_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Gold
        gold_text = self.FONT_UI.render(f"Gold: {self.player_gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (self.width - gold_text.get_width() - 10, 10))

        # Room
        room_text = self.FONT_UI.render(f"Room: {self.room_number}/{self.WIN_ROOM}", True, self.COLOR_TEXT)
        self.screen.blit(room_text, (self.width/2 - room_text.get_width()/2, 10))
        
        # Game Over / Win Text
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 50, 50)
            end_text = self.FONT_GAME_OVER.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.width/2, self.height/2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20,20))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "gold": self.player_gold,
            "room": self.room_number
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Dungeon Crawler")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        action = [0, 0, 0] # Default: no-op (defend)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            # This is a one-shot action per frame, not held.
            # We wait for a key press to submit an action.
            
            # We need to detect a key press event to advance a turn.
            # In a real game loop, we'd check for a key press and then step.
            
            # Simple manual control: wait for a key press to advance a turn
            turn_taken = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    keys_down = pygame.key.get_pressed()
                    if keys_down[pygame.K_SPACE]:
                        action = [0, 1, 0] # Attack
                    elif keys_down[pygame.K_UP]:
                        action = [1, 0, 0] # Move Up
                    elif keys_down[pygame.K_DOWN]:
                        action = [2, 0, 0] # Move Down
                    elif keys_down[pygame.K_LEFT]:
                        action = [3, 0, 0] # Move Left
                    elif keys_down[pygame.K_RIGHT]:
                        action = [4, 0, 0] # Move Right
                    else:
                        action = [0, 0, 0] # Defend
                    turn_taken = True
                    break # only one action per frame
            
            if turn_taken:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS

    env.close()