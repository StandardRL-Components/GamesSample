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



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your circle. Collect orbs of the target color shown at the bottom."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a vibrant, procedurally generated arena, collecting matching color orbs while dodging erratic enemies in a race against time."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        # Set a dummy video driver for headless operation
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_ENEMY = (255, 255, 255)
        self.ORB_COLORS = {
            "RED": (255, 50, 50),
            "GREEN": (50, 255, 50),
            "BLUE": (50, 150, 255),
            "YELLOW": (255, 255, 50),
            "PURPLE": (200, 50, 255),
        }
        self.ORB_COLOR_LIST = list(self.ORB_COLORS.values())
        
        # Game constants
        self.PLAYER_SPEED = 4
        self.PLAYER_RADIUS = 12
        self.ENEMY_RADIUS = 8
        self.ORB_RADIUS_NORMAL = 7
        self.ORB_RADIUS_LARGE = 12
        self.MAX_STEPS = 3600  # 60 seconds * 60 FPS
        self.INITIAL_LIVES = 5
        self.WIN_CONDITION_ORBS = 50
        self.NUM_ENEMIES = 5
        self.NUM_ORBS = 15
        
        # Initialize state variables
        self.player_pos = None
        self.player_lives = 0
        self.enemies = []
        self.orbs = []
        self.particles = []
        self.target_color = None
        self.score = 0
        self.steps = 0
        self.collected_orbs_count = 0
        self.enemy_speed_multiplier = 1.0
        self.last_dist_to_target = float('inf')
        self.invincibility_timer = 0
        
        # This will be properly seeded by the environment
        self.np_random = None
        
        # self.reset() # Called by the environment wrapper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_lives = self.INITIAL_LIVES
        self.invincibility_timer = 0
        
        # Game state
        self.steps = 0
        self.score = 0
        self.collected_orbs_count = 0
        self.enemy_speed_multiplier = 1.0
        
        # Entities
        self.enemies = [self._spawn_enemy() for _ in range(self.NUM_ENEMIES)]
        self.target_color = self.np_random.choice(self.ORB_COLOR_LIST)
        self.orbs = [self._spawn_orb() for _ in range(self.NUM_ORBS)]
        # Ensure at least one target orb exists
        if not any(np.array_equal(orb['color'], self.target_color) for orb in self.orbs):
            if self.orbs:
                self.orbs[0]['color'] = self.target_color
            else: # Should not happen, but for safety
                self.orbs.append(self._spawn_orb(force_color=self.target_color))

        self.particles = []
        self.last_dist_to_target = self._get_dist_to_closest_target()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement = action[0]
        reward = 0
        
        # 1. Update Player Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1  # Right
        
        if move_vec.length() > 0:
            move_vec.scale_to_length(self.PLAYER_SPEED)
            self.player_pos += move_vec
        
        # Boundary checks
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

        # 2. Update Game Logic
        self.steps += 1
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
            
        self._update_enemies()
        self._update_particles()
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 100 == 0:
            self.enemy_speed_multiplier += 0.01

        # 3. Handle Collisions and Rewards
        reward += self._handle_orb_collisions()
        if self.invincibility_timer == 0:
            reward += self._handle_enemy_collisions()
            
        # 4. Continuous Reward for moving towards target
        current_dist = self._get_dist_to_closest_target()
        dist_delta = self.last_dist_to_target - current_dist
        if current_dist < float('inf'): # Only give reward if a target exists
            if dist_delta > 0:
                reward += 0.1 * (dist_delta / self.PLAYER_SPEED) # Normalized positive reward
            else:
                reward += -0.02 # Small penalty for moving away or standing still
        self.last_dist_to_target = current_dist
        
        # 5. Check Termination Conditions
        terminated = False
        truncated = False
        if self.collected_orbs_count >= self.WIN_CONDITION_ORBS:
            reward += 100
            terminated = True
        if self.player_lives <= 0:
            reward += -100
            terminated = True
        if self.steps >= self.MAX_STEPS:
            truncated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
        
    def _spawn_enemy(self):
        # Spawn away from the center
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, 50))
        elif edge == 1: # Bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(self.HEIGHT - 50, self.HEIGHT))
        elif edge == 2: # Left
            pos = pygame.Vector2(self.np_random.uniform(0, 50), self.np_random.uniform(0, self.HEIGHT))
        else: # Right
            pos = pygame.Vector2(self.np_random.uniform(self.WIDTH - 50, self.WIDTH), self.np_random.uniform(0, self.HEIGHT))

        pattern = self.np_random.choice(['patrol', 'circle', 'homing'])
        if pattern == 'patrol':
            target = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT))
        elif pattern == 'circle':
            target = pygame.Vector2(pos.x, pos.y) # pivot point
        else: # homing
            target = self.player_pos.copy() if self.player_pos else pygame.Vector2(self.WIDTH/2, self.HEIGHT/2)
            
        return {
            "pos": pos,
            "pattern": pattern,
            "target": target,
            "angle": self.np_random.uniform(0, 2 * math.pi),
            "base_speed": self.np_random.uniform(1.0, 2.0),
        }

    def _spawn_orb(self, force_color=None):
        pos = pygame.Vector2(
            self.np_random.uniform(20, self.WIDTH - 20),
            self.np_random.uniform(20, self.HEIGHT - 20)
        )
        color = force_color if force_color is not None else self.np_random.choice(self.ORB_COLOR_LIST)
        
        is_large = self.np_random.random() < 0.1 # 10% chance for a large orb
        
        # Make orbs near enemies large for risk/reward
        for enemy in self.enemies:
            if pos.distance_to(enemy['pos']) < 75:
                is_large = True
                break
        
        return {
            "pos": pos,
            "color": color,
            "radius": self.ORB_RADIUS_LARGE if is_large else self.ORB_RADIUS_NORMAL,
            "value": 5 if is_large else 1,
            "score": 10 if is_large else 5
        }
        
    def _update_enemies(self):
        for enemy in self.enemies:
            speed = enemy['base_speed'] * self.enemy_speed_multiplier
            if enemy['pattern'] == 'patrol':
                direction = (enemy['target'] - enemy['pos'])
                if direction.length() < 20: # Reached target, pick a new one
                    enemy['target'] = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT))
                else:
                    enemy['pos'] += direction.normalize() * speed
            elif enemy['pattern'] == 'circle':
                enemy['angle'] += 0.03
                offset = pygame.Vector2(math.cos(enemy['angle']), math.sin(enemy['angle'])) * 40
                enemy['pos'] = enemy['target'] + offset
            elif enemy['pattern'] == 'homing':
                # Update target occasionally to not be perfectly accurate
                if self.steps % 30 == 0:
                    enemy['target'] = self.player_pos.copy()
                direction = (enemy['target'] - enemy['pos'])
                if direction.length() > 0:
                    enemy['pos'] += direction.normalize() * speed * 0.7 # Homing is a bit slower
            
            # Boundary check for enemies
            enemy['pos'].x = np.clip(enemy['pos'].x, self.ENEMY_RADIUS, self.WIDTH - self.ENEMY_RADIUS)
            enemy['pos'].y = np.clip(enemy['pos'].y, self.ENEMY_RADIUS, self.HEIGHT - self.ENEMY_RADIUS)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)

    def _create_particles(self, pos, color, count):
        # sfx: play_collect_sound() or play_hit_sound()
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "color": color,
                "lifetime": self.np_random.integers(20, 40),
                "radius": self.np_random.uniform(2, 5)
            })

    def _handle_orb_collisions(self):
        reward = 0
        orbs_to_remove = []
        for i, orb in enumerate(self.orbs):
            if self.player_pos.distance_to(orb['pos']) < self.PLAYER_RADIUS + orb['radius']:
                orbs_to_remove.append(i)
                self._create_particles(orb['pos'], tuple(orb['color']), 20)
                
                if np.array_equal(orb['color'], self.target_color):
                    # Correct orb
                    reward += orb['value']
                    self.score += orb['score']
                    self.collected_orbs_count += 1
                    self.target_color = self.np_random.choice(self.ORB_COLOR_LIST)
                else:
                    # Wrong orb, changes target
                    reward += 0.1
                    self.score += 1
                    self.target_color = orb['color']
        
        if orbs_to_remove:
            self.orbs = [orb for i, orb in enumerate(self.orbs) if i not in orbs_to_remove]
            for _ in range(len(orbs_to_remove)):
                self.orbs.append(self._spawn_orb())
        return reward

    def _handle_enemy_collisions(self):
        reward = 0
        for enemy in self.enemies:
            if self.player_pos.distance_to(enemy['pos']) < self.PLAYER_RADIUS + self.ENEMY_RADIUS:
                # sfx: play_damage_sound()
                self.player_lives -= 1
                reward -= 1.0 # Immediate penalty
                self.invincibility_timer = 60 # 1 second of invincibility
                self._create_particles(self.player_pos, (255, 100, 100), 30)
                # Small knockback
                knockback = (self.player_pos - enemy['pos']).normalize() * 15
                self.player_pos += knockback
                break # Only one collision per frame
        return reward

    def _get_dist_to_closest_target(self):
        target_orbs = [orb for orb in self.orbs if np.array_equal(orb['color'], self.target_color)]
        if not target_orbs:
            return float('inf')
        closest_dist = min(self.player_pos.distance_to(orb['pos']) for orb in target_orbs)
        return closest_dist

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw Orbs (background elements)
        for orb in self.orbs:
            color_tuple = tuple(orb['color'])
            # Draw a darker fill
            pygame.gfxdraw.filled_circle(
                self.screen, int(orb['pos'].x), int(orb['pos'].y), 
                int(orb['radius']), 
                tuple(c // 2 for c in color_tuple)
            )
            # Draw a bright, anti-aliased outline
            pygame.gfxdraw.aacircle(
                self.screen, int(orb['pos'].x), int(orb['pos'].y), 
                int(orb['radius']), color_tuple
            )

        # Draw Enemies
        for enemy in self.enemies:
            rect = pygame.Rect(
                enemy['pos'].x - self.ENEMY_RADIUS, 
                enemy['pos'].y - self.ENEMY_RADIUS,
                self.ENEMY_RADIUS * 2, 
                self.ENEMY_RADIUS * 2
            )
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=2)
            
        # Draw Particles
        for p in self.particles:
            color_tuple = tuple(p['color'])
            alpha_color = (*color_tuple, int(255 * (p['lifetime'] / 40.0)))
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, alpha_color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (p['pos'].x - p['radius'], p['pos'].y - p['radius']))

        # Draw Player (interactive element, bright and clear)
        player_render_pos = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Invincibility flash
        if self.invincibility_timer > 0 and (self.steps // 3) % 2 == 0:
            pass # Don't draw player to make it flash
        else:
            # Glow effect
            glow_radius = int(self.PLAYER_RADIUS * 1.8)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER, 50), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (player_render_pos[0] - glow_radius, player_render_pos[1] - glow_radius))

            # Main body
            pygame.gfxdraw.filled_circle(self.screen, player_render_pos[0], player_render_pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, player_render_pos[0], player_render_pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        target_color_tuple = tuple(self.target_color)
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, (200, 200, 220))
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 60.0
        timer_text = self.font_small.render(f"TIME: {time_left:.1f}", True, (200, 200, 220))
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Lives
        lives_text = self.font_small.render("LIVES: ", True, (200, 200, 220))
        self.screen.blit(lives_text, (10, 35))
        for i in range(self.player_lives):
            pygame.gfxdraw.filled_circle(self.screen, 70 + i * 20, 45, 6, (255, 80, 80))
            pygame.gfxdraw.aacircle(self.screen, 70 + i * 20, 45, 6, (255, 80, 80))
            
        # Collected Orbs / Goal
        goal_text = self.font_large.render(f"{self.collected_orbs_count} / {self.WIN_CONDITION_ORBS}", True, (220, 220, 240))
        goal_rect = goal_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 50))
        self.screen.blit(goal_text, goal_rect)
        
        # Target Orb Indicator
        target_text = self.font_small.render("TARGET", True, target_color_tuple)
        target_rect = target_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 20))
        self.screen.blit(target_text, target_rect)
        pygame.draw.rect(self.screen, target_color_tuple, (target_rect.left - 10, target_rect.top - 2, target_rect.width + 20, target_rect.height + 4), 2, border_radius=5)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "collected_orbs": self.collected_orbs_count
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # To run with a display, comment out the next line
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    # env.validate_implementation() # Call this after __init__ if you want to test
    
    # --- Human Play Loop ---
    # To run with a display, you need to unset the dummy video driver
    # and create a display screen.
    is_headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"

    obs, info = env.reset()
    terminated = False
    truncated = False
    
    display_screen = None
    if not is_headless:
        pygame.display.init()
        display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption(env.game_description)

    clock = pygame.time.Clock()
    action = env.action_space.sample()
    action.fill(0)

    while not terminated and not truncated:
        if not is_headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            
            # Reset action
            action.fill(0)
            
            # Movement
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            else:
                action[0] = 0 # No-op
                
            # Other actions (unused in this game)
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1
        else: # In headless mode, just take random actions
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        
        if not is_headless:
            # Render the observation to the display screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if info['steps'] % 60 == 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Orbs: {info['collected_orbs']}, Reward: {reward:.2f}")

        clock.tick(60)

    print("Game Over!")
    print(f"Final Score: {info['score']}")
    env.close()