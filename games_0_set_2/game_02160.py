
# Generated: 2025-08-27T19:27:25.185398
# Source Brief: brief_02160.md
# Brief Index: 2160

        
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


class Particle:
    """A simple particle for effects."""
    def __init__(self, x, y, color, size, life, dx, dy):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.life = life
        self.dx = dx
        self.dy = dy

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.life -= 1
        self.size = max(0, self.size - 0.2)

    def draw(self, surface):
        if self.life > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.size))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Hold space to attack in your last moved direction."
    )

    game_description = (
        "Navigate a grid, strategically battling 25 procedurally generated monsters "
        "to achieve the highest score before losing all 3 lives."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 10
        self.CELL_SIZE = 40
        self.MAX_STEPS = 1000
        self.NUM_MONSTERS = 25
        self.PLAYER_MAX_HEALTH = 3
        self.MONSTER_MAX_HEALTH = 3
        self.PLAYER_ATTACK_DAMAGE = 1
        self.MONSTER_CHASE_RADIUS = 5

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_MONSTER = (255, 50, 50)
        self.COLOR_ATTACK = (255, 255, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEART = (255, 0, 100)
        self.COLOR_HP_BG = (80, 80, 80)
        
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
        self.font_small = pygame.font.SysFont("sans", 20)
        self.font_large = pygame.font.SysFont("sans", 30)

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_last_move_dir = None
        self.player_hit_timer = None
        self.monsters = None
        self.particles = None
        self.attack_effect = None
        self.steps = None
        self.score = None
        self.game_over = None
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_last_move_dir = (0, -1)  # Default to UP
        self.player_hit_timer = 0
        
        self.monsters = []
        occupied_tiles = {self.player_pos}
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                occupied_tiles.add((self.player_pos[0] + dx, self.player_pos[1] + dy))

        while len(self.monsters) < self.NUM_MONSTERS:
            mx = self.np_random.integers(0, self.GRID_WIDTH)
            my = self.np_random.integers(0, self.GRID_HEIGHT)
            if (mx, my) not in occupied_tiles:
                self.monsters.append({
                    "pos": (mx, my),
                    "health": self.MONSTER_MAX_HEALTH,
                    "hit_timer": 0
                })
                occupied_tiles.add((mx, my))

        self.particles = []
        self.attack_effect = None # {"start": (x,y), "end": (x,y), "timer": t}
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.1  # Cost for existing
        
        movement = action[0]
        space_held = action[1] == 1
        
        # 1. Update effect timers
        self._update_effects()

        # 2. Player Action
        # Handle Movement
        moved = False
        if movement > 0:
            px, py = self.player_pos
            if movement == 1: # Up
                py = max(0, py - 1)
                self.player_last_move_dir = (0, -1)
            elif movement == 2: # Down
                py = min(self.GRID_HEIGHT - 1, py + 1)
                self.player_last_move_dir = (0, 1)
            elif movement == 3: # Left
                px = max(0, px - 1)
                self.player_last_move_dir = (-1, 0)
            elif movement == 4: # Right
                px = min(self.GRID_WIDTH - 1, px + 1)
                self.player_last_move_dir = (1, 0)
            
            if self.player_pos != (px, py):
                self.player_pos = (px, py)
                moved = True

        # Handle Attack
        if space_held:
            # sfx: player_attack_swoosh.wav
            attack_pos = self.player_pos
            for _ in range(max(self.GRID_WIDTH, self.GRID_HEIGHT)):
                attack_pos = (attack_pos[0] + self.player_last_move_dir[0], attack_pos[1] + self.player_last_move_dir[1])
                
                target_hit = False
                for monster in self.monsters:
                    if monster["pos"] == attack_pos:
                        monster["health"] -= self.PLAYER_ATTACK_DAMAGE
                        monster["hit_timer"] = 3 # Flash for 3 steps
                        target_hit = True
                        if monster["health"] <= 0:
                            reward += 10
                            self.score += 100
                            self._create_explosion(monster["pos"])
                        break
                if target_hit:
                    break
            
            # Visual effect for the attack
            start_pixel = self._grid_to_pixel(self.player_pos)
            end_pixel = self._grid_to_pixel(attack_pos)
            self.attack_effect = {"start": start_pixel, "end": end_pixel, "timer": 2}


        # 3. Monster Actions
        monsters_to_remove = []
        for i, monster in enumerate(self.monsters):
            if monster["health"] <= 0:
                monsters_to_remove.append(i)
                continue

            # AI: Chase or wander
            dist_to_player = abs(monster["pos"][0] - self.player_pos[0]) + abs(monster["pos"][1] - self.player_pos[1])
            mx, my = monster["pos"]

            if 1 < dist_to_player <= self.MONSTER_CHASE_RADIUS:
                # Chase
                if self.player_pos[0] > mx: mx += 1
                elif self.player_pos[0] < mx: mx -= 1
                elif self.player_pos[1] > my: my += 1
                elif self.player_pos[1] < my: my -= 1
            else:
                # Wander
                direction = self.np_random.integers(0, 4)
                if direction == 0: mx += 1
                elif direction == 1: mx -= 1
                elif direction == 2: my += 1
                elif direction == 3: my -= 1
            
            # Boundary check
            mx = max(0, min(self.GRID_WIDTH - 1, mx))
            my = max(0, min(self.GRID_HEIGHT - 1, my))
            monster["pos"] = (mx, my)

        # Remove dead monsters
        for i in sorted(monsters_to_remove, reverse=True):
            # sfx: monster_death_explosion.wav
            del self.monsters[i]

        # 4. Collision Check
        for monster in self.monsters:
            if monster["pos"] == self.player_pos and self.player_hit_timer == 0:
                self.player_health -= 1
                self.player_hit_timer = 5 # Invincibility for 5 steps
                reward -= 5
                # sfx: player_damage.wav
                break

        # 5. Update game state
        self.steps += 1
        
        # 6. Check termination conditions
        terminated = False
        if self.player_health <= 0:
            terminated = True
            self.game_over = True
        elif not self.monsters:
            terminated = True
            self.game_over = True
            reward += 100
            self.score += 1000 # Victory bonus
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _grid_to_pixel(self, grid_pos):
        x = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return (x, y)

    def _update_effects(self):
        if self.player_hit_timer > 0:
            self.player_hit_timer -= 1
        
        for monster in self.monsters:
            if monster["hit_timer"] > 0:
                monster["hit_timer"] -= 1

        if self.attack_effect and self.attack_effect["timer"] > 0:
            self.attack_effect["timer"] -= 1
        else:
            self.attack_effect = None

        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _create_explosion(self, grid_pos):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            dx = math.cos(angle) * speed
            dy = math.sin(angle) * speed
            color = random.choice([self.COLOR_MONSTER, (255, 150, 50), (255, 255, 100)])
            size = random.uniform(3, 7)
            life = random.randint(15, 30)
            self.particles.append(Particle(px, py, color, size, life, dx, dy))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw monsters
        for monster in self.monsters:
            self._render_monster(monster)
        
        # Draw player
        self._render_player()

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw attack effect
        if self.attack_effect:
            alpha = int(255 * (self.attack_effect["timer"] / 2.0))
            color = (*self.COLOR_ATTACK, alpha)
            temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(temp_surf, color, self.attack_effect["start"], self.attack_effect["end"], 5)
            self.screen.blit(temp_surf, (0, 0))

    def _render_player(self):
        px, py = self._grid_to_pixel(self.player_pos)
        size = self.CELL_SIZE // 2 - 4
        
        # Flash if hit
        if self.player_hit_timer > 0 and self.steps % 2 == 0:
            return

        # Draw player body
        pygame.gfxdraw.filled_circle(self.screen, px, py, size, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, size, self.COLOR_PLAYER)

        # Draw "headlight" indicating attack direction
        h_px = px + int(self.player_last_move_dir[0] * size * 0.8)
        h_py = py + int(self.player_last_move_dir[1] * size * 0.8)
        pygame.draw.circle(self.screen, (255, 255, 255), (h_px, h_py), 3)

        # Draw health bar
        bar_width = self.CELL_SIZE - 10
        bar_height = 5
        bar_x = px - bar_width // 2
        bar_y = py - size - 10
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        pygame.draw.rect(self.screen, self.COLOR_HP_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (bar_x, bar_y, int(bar_width * health_ratio), bar_height))

    def _render_monster(self, monster):
        mx, my = self._grid_to_pixel(monster["pos"])
        size = self.CELL_SIZE // 2 - 6

        color = self.COLOR_MONSTER
        if monster["hit_timer"] > 0:
            color = (255, 255, 255) # Flash white when hit

        # Draw monster body (square)
        rect = pygame.Rect(mx - size, my - size, size * 2, size * 2)
        pygame.draw.rect(self.screen, color, rect, border_radius=3)
        
        # Draw health bar
        bar_width = self.CELL_SIZE - 10
        bar_height = 4
        bar_x = mx - bar_width // 2
        bar_y = my - size - 8
        health_ratio = monster["health"] / self.MONSTER_MAX_HEALTH
        pygame.draw.rect(self.screen, self.COLOR_HP_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_MONSTER, (bar_x, bar_y, int(bar_width * health_ratio), bar_height))


    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives (Hearts)
        heart_size = 15
        for i in range(self.player_health):
            x = self.WIDTH - 20 - (i * (heart_size + 5))
            y = 20
            # Simple heart shape with polygons
            points = [
                (x, y - heart_size * 0.4),
                (x - heart_size * 0.5, y - heart_size * 0.8),
                (x - heart_size, y - heart_size * 0.4),
                (x - heart_size, y),
                (x, y + heart_size),
                (x + heart_size, y),
                (x + heart_size, y - heart_size * 0.4),
                (x + heart_size * 0.5, y - heart_size * 0.8),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_HEART)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_HEART)

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_text = "VICTORY!" if not self.monsters else "GAME OVER"
            text_surface = self.font_large.render(win_text, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "monsters_remaining": len(self.monsters)
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

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a display for manual play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print(env.user_guide)

    while not done:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Action Generation from Keyboard ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Game Loop Control ---
        # Since auto_advance is False, we control the "speed" of manual play here.
        clock.tick(10) # 10 actions per second for responsive manual play

    print(f"Game Over! Final Info: {info}")
    env.close()