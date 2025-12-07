import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:51:01.746096
# Source Brief: brief_03047.md
# Brief Index: 3047
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Trap enemies in bubbles and pop them to score points. Watch out for wind and increasingly fast foes!"
    user_guide = "Controls: ←→ to move, ↑ to jump. Press space to shoot a bubble and shift to cycle your shot power."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLATFORM_HEIGHT = 30
    MAX_STEPS = 2000
    NUM_ENEMIES = 3

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (30, 35, 50)
    COLOR_PLATFORM = (87, 58, 46)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (50, 255, 50, 40)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_TRAPPED = (255, 150, 150)
    COLOR_BUBBLE = (100, 150, 255, 128)
    COLOR_TEXT = (220, 220, 220)
    COLOR_WIND = (150, 150, 170, 100)
    
    # --- Physics ---
    GRAVITY = 0.3
    PLAYER_FORCE = 1.0
    PLAYER_JUMP_FORCE = 7.0
    PLAYER_FRICTION = 0.9
    BUBBLE_FRICTION = 0.995
    ENEMY_FRICTION = 0.95
    BUBBLE_LIFESPAN = 300 # steps
    BUBBLE_TRAP_LIFESPAN = 450 # steps
    POWER_LEVELS = [8, 12, 16]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # --- Game State Attributes ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player = {}
        self.enemies = []
        self.bubbles = []
        self.particles = []
        
        self.wind = pygame.math.Vector2(0, 0)
        self.base_enemy_speed = 1.0
        self.power_level_index = 0
        
        self.last_space_held = False
        self.last_shift_held = False

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player = {
            'pos': pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - self.PLATFORM_HEIGHT - 20),
            'vel': pygame.math.Vector2(0, 0),
            'radius': 12,
            'aim_angle': 0.0,
            'last_move_dir': pygame.math.Vector2(1, 0) # Default aim right
        }

        # Enemy state
        self.enemies = []
        for i in range(self.NUM_ENEMIES):
            self.enemies.append({
                'pos': pygame.math.Vector2(random.uniform(50, self.SCREEN_WIDTH - 50), self.SCREEN_HEIGHT - self.PLATFORM_HEIGHT - 15),
                'vel': pygame.math.Vector2(random.choice([-1, 1]) * self.base_enemy_speed, 0),
                'radius': 10,
                'trapped': False,
                'id': i
            })
        
        # Other state
        self.bubbles = []
        self.particles = []
        self.wind = pygame.math.Vector2(random.uniform(-0.05, 0.05), 0)
        self.base_enemy_speed = 1.0
        self.power_level_index = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.steps_since_trap = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.steps_since_trap += 1
        reward = 0

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Fire bubble on press
        if space_held and not self.last_space_held:
            self._fire_bubble()
            # SFX: bubble_fire.wav
        
        # Cycle power on press
        if shift_held and not self.last_shift_held:
            self.power_level_index = (self.power_level_index + 1) % len(self.POWER_LEVELS)
            # SFX: power_cycle.wav

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Update Game Logic ---
        self._update_difficulty()
        reward += self._update_player(movement)
        self._update_enemies()
        self._update_bubbles()
        self._update_particles()
        reward += self._handle_collisions()
        self._anti_softlock()

        # --- Check Termination ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS or len(self.enemies) == 0
        truncated = self.steps >= self.MAX_STEPS

        if len(self.enemies) == 0 and not self.game_over:
            reward += 10.0 # Victory bonus
            self.game_over = True
        
        if self.game_over and reward > -100: # Don't overwrite death penalty
            reward = max(reward, 0)

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    #region Update Logic
    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.base_enemy_speed += 0.05
        if self.steps > 0 and self.steps % 500 == 0:
            self.wind.x += random.uniform(-0.01, 0.01)
            self.wind.x = np.clip(self.wind.x, -0.15, 0.15)

    def _update_player(self, movement):
        acc = pygame.math.Vector2(0, self.GRAVITY)
        move_dir = pygame.math.Vector2(0, 0)

        if movement == 1: # Up (Jump)
            # Allow jump only if on or near the platform
            if self.player['pos'].y >= self.SCREEN_HEIGHT - self.PLATFORM_HEIGHT - self.player['radius'] - 5:
                self.player['vel'].y = -self.PLAYER_JUMP_FORCE
                # SFX: jump.wav
        elif movement == 2: # Down
             acc.y += self.PLAYER_FORCE * 0.5 # Limited downward force
        elif movement == 3: # Left
            acc.x = -self.PLAYER_FORCE
            move_dir.x = -1
        elif movement == 4: # Right
            acc.x = self.PLAYER_FORCE
            move_dir.x = 1
        
        if move_dir.length() > 0:
            self.player['last_move_dir'] = move_dir.normalize()

        self.player['aim_angle'] = self.player['last_move_dir'].angle_to(pygame.math.Vector2(0, -1))
        
        self.player['vel'] += acc
        self.player['vel'] *= self.PLAYER_FRICTION
        self.player['pos'] += self.player['vel']

        # Boundary checks
        self.player['pos'].x = np.clip(self.player['pos'].x, self.player['radius'], self.SCREEN_WIDTH - self.player['radius'])
        if self.player['pos'].y > self.SCREEN_HEIGHT - self.PLATFORM_HEIGHT - self.player['radius']:
            self.player['pos'].y = self.SCREEN_HEIGHT - self.PLATFORM_HEIGHT - self.player['radius']
            self.player['vel'].y = 0
        
        return 0

    def _update_enemies(self):
        for enemy in self.enemies:
            if not enemy['trapped']:
                acc = pygame.math.Vector2(0, self.GRAVITY)
                enemy['vel'] += acc
                enemy['vel'].x *= self.ENEMY_FRICTION
                enemy['pos'] += enemy['vel']
                
                # Patrol behavior
                if enemy['pos'].x <= enemy['radius'] or enemy['pos'].x >= self.SCREEN_WIDTH - enemy['radius']:
                    enemy['vel'].x *= -1

                # Platform collision
                if enemy['pos'].y > self.SCREEN_HEIGHT - self.PLATFORM_HEIGHT - enemy['radius']:
                    enemy['pos'].y = self.SCREEN_HEIGHT - self.PLATFORM_HEIGHT - enemy['radius']
                    enemy['vel'].y = 0
                    if abs(enemy['vel'].x) < 0.1: # Start moving if stopped
                         enemy['vel'].x = random.choice([-1, 1]) * self.base_enemy_speed
    
    def _update_bubbles(self):
        for bubble in self.bubbles[:]:
            bubble['life'] -= 1
            if bubble['life'] <= 0:
                self._create_pop_particles(bubble['pos'])
                if bubble['trapped_enemy_id'] is not None:
                    # Find the enemy and un-trap it
                    for enemy in self.enemies:
                        if enemy['id'] == bubble['trapped_enemy_id']:
                            enemy['trapped'] = False
                            enemy['vel'] = pygame.math.Vector2(random.uniform(-1, 1), -2)
                            break
                self.bubbles.remove(bubble)
                # SFX: bubble_pop_fade.wav
                continue

            acc = pygame.math.Vector2(self.wind.x, self.GRAVITY * -0.1) # Bubbles are floaty
            bubble['vel'] += acc
            bubble['vel'] *= self.BUBBLE_FRICTION
            bubble['pos'] += bubble['vel']

            # Boundary bouncing
            if bubble['pos'].x <= bubble['radius'] or bubble['pos'].x >= self.SCREEN_WIDTH - bubble['radius']:
                bubble['vel'].x *= -0.8
            if bubble['pos'].y <= bubble['radius'] or bubble['pos'].y >= self.SCREEN_HEIGHT - bubble['radius']:
                bubble['vel'].y *= -0.8
            
            bubble['pos'].x = np.clip(bubble['pos'].x, bubble['radius'], self.SCREEN_WIDTH - bubble['radius'])
            bubble['pos'].y = np.clip(bubble['pos'].y, bubble['radius'], self.SCREEN_HEIGHT - bubble['radius'])

            if bubble['trapped_enemy_id'] is not None:
                for enemy in self.enemies:
                    if enemy['id'] == bubble['trapped_enemy_id']:
                        enemy['pos'] = bubble['pos']
                        break
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        # Player vs Enemies
        for enemy in self.enemies:
            if not enemy['trapped']:
                dist = self.player['pos'].distance_to(enemy['pos'])
                if dist < self.player['radius'] + enemy['radius']:
                    self.game_over = True
                    reward = -100.0
                    self._create_pop_particles(self.player['pos'], count=50, color=(255,0,0))
                    # SFX: player_death.wav
                    return reward

        # Bubbles vs Enemies
        for bubble in self.bubbles:
            if bubble['trapped_enemy_id'] is None: # Only non-trapping bubbles can trap
                for enemy in self.enemies:
                    if not enemy['trapped']:
                        dist = bubble['pos'].distance_to(enemy['pos'])
                        if dist < bubble['radius'] + enemy['radius']:
                            enemy['trapped'] = True
                            bubble['trapped_enemy_id'] = enemy['id']
                            bubble['life'] = self.BUBBLE_TRAP_LIFESPAN
                            reward += 0.1
                            self.steps_since_trap = 0
                            # SFX: enemy_trap.wav
                            break
        
        # Player vs Bubbles (for popping)
        for bubble in self.bubbles[:]:
            if bubble['trapped_enemy_id'] is not None:
                dist = self.player['pos'].distance_to(bubble['pos'])
                if dist < self.player['radius'] + bubble['radius']:
                    self._create_pop_particles(bubble['pos'], count=30, color=(255,255,100))
                    self.bubbles.remove(bubble)
                    
                    # Remove the corresponding enemy
                    enemy_to_remove = None
                    for enemy in self.enemies:
                        if enemy['id'] == bubble['trapped_enemy_id']:
                            enemy_to_remove = enemy
                            break
                    if enemy_to_remove:
                        self.enemies.remove(enemy_to_remove)

                    reward += 1.0
                    self.score += 1
                    # SFX: enemy_pop.wav
        
        return reward
    
    def _anti_softlock(self):
        # If no enemies have been trapped for a while, nudge one towards the player
        if self.steps_since_trap > 500 and len(self.enemies) > 0:
            target_enemy = random.choice([e for e in self.enemies if not e['trapped']])
            if target_enemy:
                nudge_dir = (self.player['pos'] - target_enemy['pos']).normalize()
                target_enemy['vel'] += nudge_dir * 0.5
                self.steps_since_trap = 0 # Reset timer

    def _fire_bubble(self):
        power = self.POWER_LEVELS[self.power_level_index]
        aim_vec = pygame.math.Vector2()
        aim_vec.from_polar((1, self.player['aim_angle'] - 90))
        
        bubble_vel = aim_vec * power
        bubble_pos = self.player['pos'] + aim_vec * (self.player['radius'] + 16)
        
        self.bubbles.append({
            'pos': bubble_pos,
            'vel': bubble_vel,
            'radius': 15,
            'life': self.BUBBLE_LIFESPAN,
            'trapped_enemy_id': None
        })
    
    def _create_pop_particles(self, pos, count=20, color=None):
        for _ in range(count):
            angle = random.uniform(0, 360)
            speed = random.uniform(1, 4)
            vel = pygame.math.Vector2()
            vel.from_polar((speed, angle))
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': random.randint(15, 30),
                'radius': random.uniform(1, 3),
                'color': color or (200, 220, 255)
            })
    #endregion

    #region Rendering
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_wind_particles()
        self._render_bubbles()
        self._render_enemies()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Grid
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))
        # Platform
        platform_rect = pygame.Rect(0, self.SCREEN_HEIGHT - self.PLATFORM_HEIGHT, self.SCREEN_WIDTH, self.PLATFORM_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, platform_rect)

    def _render_player(self):
        if self.game_over and self.score < self.NUM_ENEMIES: return
        
        # Glow effect
        glow_radius = int(self.player['radius'] * 2.5)
        pygame.gfxdraw.filled_circle(
            self.screen, 
            int(self.player['pos'].x), int(self.player['pos'].y), 
            glow_radius, 
            self.COLOR_PLAYER_GLOW
        )
        # Player body
        pygame.gfxdraw.filled_circle(
            self.screen, 
            int(self.player['pos'].x), int(self.player['pos'].y), 
            self.player['radius'], 
            self.COLOR_PLAYER
        )
        pygame.gfxdraw.aacircle(
            self.screen, 
            int(self.player['pos'].x), int(self.player['pos'].y), 
            self.player['radius'], 
            self.COLOR_PLAYER
        )
        # Launcher
        aim_vec = pygame.math.Vector2()
        aim_vec.from_polar((1, self.player['aim_angle'] - 90))
        start_pos = self.player['pos'] + aim_vec * self.player['radius']
        end_pos = start_pos + aim_vec * 20
        pygame.draw.line(self.screen, self.COLOR_TEXT, start_pos, end_pos, 3)

    def _render_enemies(self):
        for enemy in self.enemies:
            color = self.COLOR_ENEMY_TRAPPED if enemy['trapped'] else self.COLOR_ENEMY
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], enemy['radius'], color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], enemy['radius'], color)

    def _render_bubbles(self):
        for bubble in self.bubbles:
            pos = (int(bubble['pos'].x), int(bubble['pos'].y))
            radius = bubble['radius']
            
            # Draw bubble body with transparency
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_BUBBLE)
            
            # Draw highlight
            highlight_pos = (pos[0] + radius // 3, pos[1] - radius // 3)
            pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], radius // 4, (255, 255, 255, 60))

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], p['radius'])

    def _render_wind_particles(self):
        if 'wind_particles' not in self.__dict__:
            self.wind_particles = [{'pos': pygame.math.Vector2(random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT)), 'life': random.randint(50,100)} for _ in range(20)]
        
        for p in self.wind_particles:
            p['pos'].x += self.wind.x * 20 + random.uniform(-0.1, 0.1)
            p['pos'].y += self.wind.y * 20 + random.uniform(-0.1, 0.1)
            p['life'] -= 1
            if p['pos'].x > self.SCREEN_WIDTH: p['pos'].x = 0
            if p['pos'].x < 0: p['pos'].x = self.SCREEN_WIDTH
            if p['life'] <= 0:
                p['pos'] = pygame.math.Vector2(random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT))
                p['life'] = random.randint(50,100)
            
            end_pos = p['pos'] + self.wind * 10
            pygame.draw.line(self.screen, self.COLOR_WIND, p['pos'], end_pos, 1)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        # Enemies remaining
        enemy_text = self.font_small.render(f"ENEMIES: {len(self.enemies)}", True, self.COLOR_TEXT)
        self.screen.blit(enemy_text, (10, 10))

        # Power Meter
        power_text = self.font_small.render("POWER", True, self.COLOR_TEXT)
        self.screen.blit(power_text, (10, self.SCREEN_HEIGHT - self.PLATFORM_HEIGHT - 60))
        
        meter_width = 100
        meter_height = 10
        pygame.draw.rect(self.screen, self.COLOR_GRID, (10, self.SCREEN_HEIGHT - self.PLATFORM_HEIGHT - 40, meter_width, meter_height))
        
        current_power_width = (self.power_level_index + 1) / len(self.POWER_LEVELS) * meter_width
        power_color = [(255, 255, 0), (255, 165, 0), (255, 69, 0)][self.power_level_index]
        pygame.draw.rect(self.screen, power_color, (10, self.SCREEN_HEIGHT - self.PLATFORM_HEIGHT - 40, current_power_width, meter_height))
    #endregion

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "enemies_remaining": len(self.enemies),
            "wind_x": self.wind.x,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver to see the game
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bubble Physics Gym Environment")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Input Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0

        # --- Render to Screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()