
# Generated: 2025-08-27T19:47:42.380749
# Source Brief: brief_02258.md
# Brief Index: 2258

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move. Hold Space to fire. Avoid enemy projectiles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a robot through three stages of increasing difficulty, blasting enemies in a top-down arena shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.ARENA_MARGIN = 10

        # --- Player Constants ---
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 4
        self.MAX_PLAYER_HEALTH = 3
        self.PLAYER_SHOOT_COOLDOWN = 8  # Frames between shots
        self.PLAYER_INVINCIBILITY_FRAMES = 90 # 3 seconds at 30fps

        # --- Enemy Constants ---
        self.ENEMY_SIZE = 18
        self.ENEMY_BASE_SPEED = 1.0
        self.ENEMY_BASE_FIRE_RATE = 120 # Lower is faster

        # --- Projectile Constants ---
        self.PROJ_SPEED = 8
        self.PROJ_SIZE = (4, 12)

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_ARENA = (30, 35, 50)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_ENEMY = (80, 120, 255)
        self.COLOR_PLAYER_PROJ = (255, 255, 100)
        self.COLOR_ENEMY_PROJ = (255, 100, 200)
        self.COLOR_HEALTH_FG = (100, 255, 100)
        self.COLOR_HEALTH_BG = (60, 60, 60)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_stage = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.player_pos = [0, 0]
        self.player_health = 0
        self.player_facing_dir = [0, -1]
        self.player_shoot_cooldown = 0
        self.player_hit_cooldown = 0
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.explosions = []

        self.reset()
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        
        self.player_pos = [self.WIDTH / 2, self.HEIGHT - self.PLAYER_SIZE * 2]
        self.player_health = self.MAX_PLAYER_HEALTH
        self.player_facing_dir = [0, -1] # Default to 'up'
        self.player_shoot_cooldown = 0
        self.player_hit_cooldown = 0
        
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.explosions = []
        
        self._spawn_enemies_for_stage()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.01 # Small survival reward per step

        # --- Update Cooldowns ---
        if self.player_shoot_cooldown > 0: self.player_shoot_cooldown -= 1
        if self.player_hit_cooldown > 0: self.player_hit_cooldown -= 1

        # --- Handle Actions ---
        self._handle_player_action(action)

        # --- Update Game Logic ---
        self._update_projectiles()
        self._update_enemies()
        self._update_explosions()

        # --- Handle Collisions and Events ---
        reward += self._handle_collisions()

        # --- Check Stage Progression ---
        if not self.enemies:
            reward += 100 # Stage clear reward
            self.stage += 1
            if self.stage > 3:
                self.game_over = True
                reward += 500 # Win game reward
            else:
                self._spawn_enemies_for_stage()
                # SFX: stage_clear

        # --- Check Termination Conditions ---
        terminated = self.game_over
        if self.player_health <= 0 and not terminated:
            self.game_over = True
            terminated = True
            reward -= 100 # Lose game penalty
            # SFX: game_over
        
        if self.steps >= self.MAX_STEPS and not terminated:
            self.game_over = True
            terminated = True

        self.steps += 1
        self.clock.tick(30) # Maintain 30 FPS for auto-advance

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        move_vec = [0, 0]
        if movement == 1: move_vec[1] = -1 # Up
        elif movement == 2: move_vec[1] = 1 # Down
        elif movement == 3: move_vec[0] = -1 # Left
        elif movement == 4: move_vec[0] = 1 # Right

        if movement != 0:
            self.player_facing_dir = move_vec
            self.player_pos[0] += move_vec[0] * self.PLAYER_SPEED
            self.player_pos[1] += move_vec[1] * self.PLAYER_SPEED

        # Clamp player position
        self.player_pos[0] = max(self.ARENA_MARGIN + self.PLAYER_SIZE/2, min(self.WIDTH - self.ARENA_MARGIN - self.PLAYER_SIZE/2, self.player_pos[0]))
        self.player_pos[1] = max(self.ARENA_MARGIN + self.PLAYER_SIZE/2, min(self.HEIGHT - self.ARENA_MARGIN - self.PLAYER_SIZE/2, self.player_pos[1]))

        if space_held and self.player_shoot_cooldown == 0:
            self.player_shoot_cooldown = self.PLAYER_SHOOT_COOLDOWN
            proj_pos = list(self.player_pos)
            self.player_projectiles.append({"pos": proj_pos, "dir": self.player_facing_dir})
            # SFX: player_shoot

    def _spawn_enemies_for_stage(self):
        stage_configs = {
            1: {"count": 5, "speed_mult": 1.0, "fire_mult": 1.0},
            2: {"count": 10, "speed_mult": 1.5, "fire_mult": 0.75},
            3: {"count": 15, "speed_mult": 2.0, "fire_mult": 0.5},
        }
        config = stage_configs[self.stage]
        
        for _ in range(config["count"]):
            side = self.np_random.integers(4)
            if side == 0: x, y = self.np_random.uniform(self.ARENA_MARGIN, self.WIDTH - self.ARENA_MARGIN), self.ARENA_MARGIN + self.ENEMY_SIZE
            elif side == 1: x, y = self.WIDTH - self.ARENA_MARGIN - self.ENEMY_SIZE, self.np_random.uniform(self.ARENA_MARGIN, self.HEIGHT - self.ARENA_MARGIN)
            elif side == 2: x, y = self.np_random.uniform(self.ARENA_MARGIN, self.WIDTH - self.ARENA_MARGIN), self.HEIGHT - self.ARENA_MARGIN - self.ENEMY_SIZE
            else: x, y = self.ARENA_MARGIN + self.ENEMY_SIZE, self.np_random.uniform(self.ARENA_MARGIN, self.HEIGHT - self.ARENA_MARGIN)

            pattern_type = self.np_random.choice(['patrol_h', 'patrol_v', 'circle'])
            
            self.enemies.append({
                "pos": [x, y],
                "speed": self.ENEMY_BASE_SPEED * config["speed_mult"],
                "fire_rate": int(self.ENEMY_BASE_FIRE_RATE * config["fire_mult"]),
                "fire_cooldown": self.np_random.integers(0, int(self.ENEMY_BASE_FIRE_RATE * config["fire_mult"])),
                "pattern": pattern_type,
                "patrol_dir": 1,
                "angle": self.np_random.uniform(0, 2 * math.pi) # For circular motion
            })

    def _update_enemies(self):
        for enemy in self.enemies:
            # Movement
            if enemy['pattern'] == 'patrol_h':
                enemy['pos'][0] += enemy['speed'] * enemy['patrol_dir']
                if enemy['pos'][0] > self.WIDTH - self.ARENA_MARGIN - self.ENEMY_SIZE or enemy['pos'][0] < self.ARENA_MARGIN + self.ENEMY_SIZE:
                    enemy['patrol_dir'] *= -1
            elif enemy['pattern'] == 'patrol_v':
                enemy['pos'][1] += enemy['speed'] * enemy['patrol_dir']
                if enemy['pos'][1] > self.HEIGHT - self.ARENA_MARGIN - self.ENEMY_SIZE or enemy['pos'][1] < self.ARENA_MARGIN + self.ENEMY_SIZE:
                    enemy['patrol_dir'] *= -1
            elif enemy['pattern'] == 'circle':
                center_x, center_y = self.WIDTH / 2, self.HEIGHT / 2
                radius = min(self.WIDTH, self.HEIGHT) / 4
                enemy['angle'] += 0.02 * enemy['speed']
                enemy['pos'][0] = center_x + math.cos(enemy['angle']) * radius
                enemy['pos'][1] = center_y + math.sin(enemy['angle']) * radius

            # Shooting
            enemy['fire_cooldown'] -= 1
            if enemy['fire_cooldown'] <= 0:
                enemy['fire_cooldown'] = enemy['fire_rate']
                dx = self.player_pos[0] - enemy['pos'][0]
                dy = self.player_pos[1] - enemy['pos'][1]
                dist = math.hypot(dx, dy)
                if dist > 0:
                    direction = [dx / dist, dy / dist]
                    self.enemy_projectiles.append({"pos": list(enemy["pos"]), "dir": direction})
                    # SFX: enemy_shoot

    def _update_projectiles(self):
        for proj in self.player_projectiles:
            proj['pos'][0] += proj['dir'][0] * self.PROJ_SPEED
            proj['pos'][1] += proj['dir'][1] * self.PROJ_SPEED
        for proj in self.enemy_projectiles:
            proj['pos'][0] += proj['dir'][0] * self.PROJ_SPEED
            proj['pos'][1] += proj['dir'][1] * self.PROJ_SPEED
        
        # Remove off-screen projectiles
        self.player_projectiles = [p for p in self.player_projectiles if self._is_on_screen(p['pos'])]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if self._is_on_screen(p['pos'])]

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                if math.hypot(proj['pos'][0] - enemy['pos'][0], proj['pos'][1] - enemy['pos'][1]) < (self.PROJ_SIZE[1]/2 + self.ENEMY_SIZE/2):
                    self.player_projectiles.remove(proj)
                    self.enemies.remove(enemy)
                    self.score += 10
                    reward += 10
                    self._create_explosion(enemy['pos'])
                    # SFX: enemy_hit
                    break
        
        # Enemy projectiles vs player
        if self.player_hit_cooldown == 0:
            for proj in self.enemy_projectiles[:]:
                if math.hypot(proj['pos'][0] - self.player_pos[0], proj['pos'][1] - self.player_pos[1]) < (self.PROJ_SIZE[1]/2 + self.PLAYER_SIZE/2):
                    self.enemy_projectiles.remove(proj)
                    self.player_health -= 1
                    self.player_hit_cooldown = self.PLAYER_INVINCIBILITY_FRAMES
                    reward -= 1
                    self._create_explosion(self.player_pos, large=True)
                    # SFX: player_hit
                    break
        return reward

    def _create_explosion(self, pos, large=False):
        self.explosions.append({
            "pos": list(pos),
            "radius": 0,
            "max_radius": 30 if large else 20,
            "duration": 15 if large else 10,
            "life": 15 if large else 10,
        })

    def _update_explosions(self):
        for exp in self.explosions:
            exp['life'] -= 1
            exp['radius'] = exp['max_radius'] * (1 - exp['life'] / exp['duration'])
        self.explosions = [exp for exp in self.explosions if exp['life'] > 0]

    def _is_on_screen(self, pos):
        return 0 < pos[0] < self.WIDTH and 0 < pos[1] < self.HEIGHT

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Arena
        pygame.draw.rect(self.screen, self.COLOR_ARENA, (self.ARENA_MARGIN, self.ARENA_MARGIN, self.WIDTH - 2*self.ARENA_MARGIN, self.HEIGHT - 2*self.ARENA_MARGIN))
        
        # Enemies
        for enemy in self.enemies:
            x, y = int(enemy['pos'][0]), int(enemy['pos'][1])
            pygame.gfxdraw.aacircle(self.screen, x, y, int(self.ENEMY_SIZE/2), self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, x, y, int(self.ENEMY_SIZE/2), self.COLOR_ENEMY)

        # Player Projectiles
        for proj in self.player_projectiles:
            x, y = proj['pos']
            dx, dy = proj['dir']
            angle = math.degrees(math.atan2(-dx, -dy))
            surf = pygame.Surface(self.PROJ_SIZE, pygame.SRCALPHA)
            surf.fill(self.COLOR_PLAYER_PROJ)
            rotated_surf = pygame.transform.rotate(surf, angle)
            rect = rotated_surf.get_rect(center=(int(x), int(y)))
            self.screen.blit(rotated_surf, rect)

        # Enemy Projectiles
        for proj in self.enemy_projectiles:
            x, y = int(proj['pos'][0]), int(proj['pos'][1])
            pygame.gfxdraw.aacircle(self.screen, x, y, int(self.PROJ_SIZE[0]/2)+1, self.COLOR_ENEMY_PROJ)
            pygame.gfxdraw.filled_circle(self.screen, x, y, int(self.PROJ_SIZE[0]/2)+1, self.COLOR_ENEMY_PROJ)

        # Player
        is_invincible = self.player_hit_cooldown > 0
        if not (is_invincible and (self.steps // 3) % 2 == 0):
            player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
            player_rect.center = (int(self.player_pos[0]), int(self.player_pos[1]))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
            
        # Explosions
        for exp in self.explosions:
            alpha = 255 * (exp['life'] / exp['duration'])
            color = (*self.COLOR_WHITE, alpha)
            surf = pygame.Surface((exp['max_radius']*2, exp['max_radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (exp['max_radius'], exp['max_radius']), int(exp['radius']))
            self.screen.blit(surf, (int(exp['pos'][0] - exp['max_radius']), int(exp['pos'][1] - exp['max_radius'])))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 15))

        # Health Bar
        health_bar_width = 150
        health_bar_height = 20
        health_ratio = max(0, self.player_health / self.MAX_PLAYER_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (15, 15, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (15, 15, health_bar_width * health_ratio, health_bar_height))
        
        # Stage Display
        stage_text_str = f"STAGE {self.stage}"
        if self.game_over:
            stage_text_str = "GAME OVER" if self.player_health <= 0 else "YOU WIN!"

        stage_text = self.font_stage.render(stage_text_str, True, self.COLOR_TEXT)
        text_rect = stage_text.get_rect(center=(self.WIDTH/2, self.HEIGHT - 30))
        self.screen.blit(stage_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "player_health": self.player_health,
            "enemies_left": len(self.enemies),
        }
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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

# Example usage to run and display the game
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for display ---
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Robot Arena")
    running = True
    
    while running:
        action = [0, 0, 0] # Default no-op
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Blit the observation from the env to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()

    env.close()