
# Generated: 2025-08-28T00:36:39.041070
# Source Brief: brief_03838.md
# Brief Index: 3838

        
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
        "Controls: ↑↓ to aim. Press Space to shoot. Press Shift to reload."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a zombie horde by strategically shooting and reloading in this side-view shooter. Eliminate all zombies to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        
        # Player
        self.MAX_HEALTH = 100
        self.MAX_AMMO = 30
        self.PLAYER_POS = (50, self.HEIGHT // 2 + 100)
        self.PLAYER_SIZE = (12, 24)
        self.AIM_SPEED = 2.0  # Degrees per step
        self.MIN_AIM_ANGLE = -60
        self.MAX_AIM_ANGLE = 60

        # Zombies
        self.TOTAL_ZOMBIES = 25
        self.ZOMBIE_SIZE = (14, 28)
        self.ZOMBIE_SPEED = 4
        self.ZOMBIE_DAMAGE = 10
        
        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GROUND = (40, 45, 50)
        self.COLOR_PLAYER = (255, 65, 54)
        self.COLOR_ZOMBIE = (46, 204, 64)
        self.COLOR_BULLET_TRACE = (255, 220, 0, 150)
        self.COLOR_MUZZLE_FLASH = (255, 255, 200)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR = (46, 204, 64)
        self.COLOR_HEALTH_BAR_BG = (100, 20, 20)
        self.COLOR_HIT_SPARK = (255, 180, 0)
        self.COLOR_RELOAD = (0, 116, 217)
        self.COLOR_EMPTY = (255, 133, 27)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_health = 0
        self.player_ammo = 0
        self.player_aim_angle = 0.0
        self.zombies = []
        self.zombies_killed = 0
        self.last_action_feedback = {}

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_health = self.MAX_HEALTH
        self.player_ammo = self.MAX_AMMO
        self.player_aim_angle = 0.0
        
        self.zombies_killed = 0
        self.zombies = self._spawn_zombies()
        
        self.last_action_feedback.clear()
        
        return self._get_observation(), self._get_info()

    def _spawn_zombies(self):
        zombies = []
        for _ in range(self.TOTAL_ZOMBIES):
            zombies.append({
                'pos': [
                    self.WIDTH + self.np_random.integers(50, 800),
                    self.np_random.integers(self.HEIGHT - 80, self.HEIGHT - self.ZOMBIE_SIZE[1] // 2)
                ],
                'alive': True,
                'move_counter': self.np_random.integers(0, 3)
            })
        return zombies
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        self.last_action_feedback.clear()

        # 1. Handle Player Actions
        reloaded_this_step = False
        if shift_held:
            # // SFX: Reload start
            self.last_action_feedback['type'] = 'reload'
            if self.player_ammo < self.MAX_AMMO:
                self.player_ammo = self.MAX_AMMO
                reloaded_this_step = True
                # // SFX: Reload complete
        elif space_held and not reloaded_this_step:
            if self.player_ammo > 0:
                self.player_ammo -= 1
                self.last_action_feedback['type'] = 'shoot'
                # // SFX: Gunshot
                
                hit, hit_pos, hit_zombie_index = self._raycast_shot()
                if hit:
                    self.zombies[hit_zombie_index]['alive'] = False
                    self.zombies_killed += 1
                    reward += 1.1  # +1 for kill, +0.1 for hit
                    self.last_action_feedback['hit_pos'] = hit_pos
                    # // SFX: Zombie hit/death
                else:
                    reward -= 0.01  # Miss penalty
                    # // SFX: Bullet whiz/miss
            else:
                self.last_action_feedback['type'] = 'empty'
                # // SFX: Empty click
        
        if movement == 1:  # Up
            self.player_aim_angle = max(self.MIN_AIM_ANGLE, self.player_aim_angle - self.AIM_SPEED)
        elif movement == 2:  # Down
            self.player_aim_angle = min(self.MAX_AIM_ANGLE, self.player_aim_angle + self.AIM_SPEED)

        # 2. Update World State (Zombies)
        player_rect = pygame.Rect(self.PLAYER_POS[0] - self.PLAYER_SIZE[0] // 2, self.PLAYER_POS[1] - self.PLAYER_SIZE[1], *self.PLAYER_SIZE)
        
        for zombie in self.zombies:
            if not zombie['alive']:
                continue

            # Movement
            zombie['move_counter'] = (zombie['move_counter'] + 1) % 3
            if zombie['move_counter'] != 0:
                zombie['pos'][0] -= self.ZOMBIE_SPEED
            else:
                zombie['pos'][0] += self.ZOMBIE_SPEED / 2
            
            # Collision with player
            zombie_rect = pygame.Rect(zombie['pos'][0] - self.ZOMBIE_SIZE[0] // 2, zombie['pos'][1] - self.ZOMBIE_SIZE[1], *self.ZOMBIE_SIZE)
            if player_rect.colliderect(zombie_rect):
                self.player_health -= self.ZOMBIE_DAMAGE
                reward -= 1.0
                zombie['alive'] = False
                self.zombies_killed += 1 # A zombie that attacks is "dealt with"
                self.last_action_feedback['player_hit'] = True
                # // SFX: Player hurt
                break # One hit per step

        # 3. Finalize Step
        self.steps += 1
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            self.win = False
            return True, -100.0
        if self.zombies_killed >= self.TOTAL_ZOMBIES:
            self.game_over = True
            self.win = True
            return True, 100.0
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            return True, 0.0
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "zombies_left": self.TOTAL_ZOMBIES - self.zombies_killed}

    def _render_game(self):
        # Ground
        ground_y = self.HEIGHT - 40
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, ground_y, self.WIDTH, 40))

        # Zombies
        for zombie in self.zombies:
            if zombie['alive']:
                z_rect = pygame.Rect(zombie['pos'][0] - self.ZOMBIE_SIZE[0] // 2, zombie['pos'][1] - self.ZOMBIE_SIZE[1], *self.ZOMBIE_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z_rect)

        # Player
        p_rect = pygame.Rect(self.PLAYER_POS[0] - self.PLAYER_SIZE[0] // 2, self.PLAYER_POS[1] - self.PLAYER_SIZE[1], *self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, p_rect)
        
        # Aiming line
        rad_angle = math.radians(self.player_aim_angle)
        gun_barrel_pos = (self.PLAYER_POS[0] + 5, self.PLAYER_POS[1] - 12)
        end_pos = (gun_barrel_pos[0] + self.WIDTH, gun_barrel_pos[1] + self.WIDTH * math.tan(rad_angle))
        pygame.draw.aaline(self.screen, (255, 255, 255, 20), gun_barrel_pos, end_pos)
        
        # Action feedback
        feedback = self.last_action_feedback
        if feedback.get('player_hit'):
            self.screen.fill((255, 0, 0), special_flags=pygame.BLEND_RGB_ADD)

        if feedback.get('type') == 'shoot':
            # Muzzle flash
            flash_points = []
            for i in range(5):
                angle_offset = math.radians(self.player_aim_angle + self.np_random.uniform(-15, 15))
                length = self.np_random.uniform(15, 25)
                end_pt = (gun_barrel_pos[0] + length * math.cos(angle_offset), gun_barrel_pos[1] + length * math.sin(angle_offset))
                pygame.draw.line(self.screen, self.COLOR_MUZZLE_FLASH, gun_barrel_pos, end_pt, 3)
            
            # Bullet trace
            hit_pos = feedback.get('hit_pos')
            if hit_pos:
                pygame.draw.line(self.screen, self.COLOR_BULLET_TRACE, gun_barrel_pos, hit_pos, 2)
                # Hit sparks
                for _ in range(10):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    length = self.np_random.uniform(2, 8)
                    spark_end = (hit_pos[0] + length * math.cos(angle), hit_pos[1] + length * math.sin(angle))
                    pygame.draw.line(self.screen, self.COLOR_HIT_SPARK, hit_pos, spark_end, 1)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.MAX_HEALTH)
        bar_width = 150
        pygame.gfxdraw.box(self.screen, pygame.Rect(10, 10, bar_width, 20), (*self.COLOR_HEALTH_BAR_BG, 180))
        if health_ratio > 0:
            pygame.gfxdraw.box(self.screen, pygame.Rect(10, 10, int(bar_width * health_ratio), 20), self.COLOR_HEALTH_BAR)
        pygame.gfxdraw.rectangle(self.screen, pygame.Rect(10, 10, bar_width, 20), (*self.COLOR_UI_TEXT, 100))

        # Ammo Counter
        ammo_text = f"AMMO: {self.player_ammo}/{self.MAX_AMMO}"
        self._render_text(ammo_text, (self.WIDTH - 10, 10), self.font_small, self.COLOR_UI_TEXT, align="topright")

        # Zombies Remaining
        zombie_text = f"REMAINING: {self.TOTAL_ZOMBIES - self.zombies_killed}"
        self._render_text(zombie_text, (self.WIDTH // 2, 10), self.font_small, self.COLOR_UI_TEXT, align="midtop")

        # Action text
        feedback = self.last_action_feedback
        if feedback.get('type') == 'reload':
             self._render_text("RELOADING", (self.PLAYER_POS[0], self.PLAYER_POS[1] - 50), self.font_small, self.COLOR_RELOAD)
        elif feedback.get('type') == 'empty':
             self._render_text("EMPTY", (self.PLAYER_POS[0], self.PLAYER_POS[1] - 50), self.font_small, self.COLOR_EMPTY)

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.win:
                self._render_text("VICTORY", (self.WIDTH // 2, self.HEIGHT // 2 - 30), self.font_large, self.COLOR_ZOMBIE)
            else:
                self._render_text("YOU DIED", (self.WIDTH // 2, self.HEIGHT // 2 - 30), self.font_large, self.COLOR_PLAYER)
            self._render_text(f"SCORE: {int(self.score)}", (self.WIDTH // 2, self.HEIGHT // 2 + 30), self.font_medium, self.COLOR_UI_TEXT)

    def _render_text(self, text, pos, font, color, align="center"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "topleft":
            text_rect.topleft = pos
        elif align == "topright":
            text_rect.topright = pos
        elif align == "midtop":
            text_rect.midtop = pos
        self.screen.blit(text_surface, text_rect)

    def _raycast_shot(self):
        rad_angle = math.radians(self.player_aim_angle)
        gun_barrel_pos = (self.PLAYER_POS[0] + 5, self.PLAYER_POS[1] - 12)

        # Get alive zombies sorted by distance
        sorted_zombies = sorted(
            [(i, z) for i, z in enumerate(self.zombies) if z['alive']],
            key=lambda item: item[1]['pos'][0]
        )

        for i, zombie in sorted_zombies:
            z_pos = zombie['pos']
            z_rect = pygame.Rect(z_pos[0] - self.ZOMBIE_SIZE[0] // 2, z_pos[1] - self.ZOMBIE_SIZE[1], *self.ZOMBIE_SIZE)

            dist_to_zombie = z_rect.centerx - gun_barrel_pos[0]
            if dist_to_zombie <= 0:
                continue

            ray_y_at_zombie_x = gun_barrel_pos[1] + dist_to_zombie * math.tan(rad_angle)
            
            if z_rect.top <= ray_y_at_zombie_x <= z_rect.bottom:
                hit_pos = (z_rect.centerx, int(ray_y_at_zombie_x))
                return True, hit_pos, i

        return False, None, None
    
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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Set up Pygame window for human play
    pygame.display.set_caption("Zombie Horde")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0]) # No-op
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")
    
    running = True
    while running:
        # --- Human Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Movement (Aim)
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        # Space (Shoot)
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        # Shift (Reload)
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Zombies: {info['zombies_left']}")

        if terminated or truncated:
            print("\n--- GAME OVER ---")
            print(f"Final Score: {info['score']:.2f}")
            print("Resetting in 3 seconds...")
            
            # Display final frame
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            pygame.time.wait(3000)
            obs, info = env.reset()

        # --- Render ---
        # The observation is the frame, so we just need to display it
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Since auto_advance is False, we control the step rate
        clock.tick(10) # Run at 10 steps per second for human playability

    env.close()