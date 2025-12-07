
# Generated: 2025-08-27T18:39:17.506691
# Source Brief: brief_01902.md
# Brief Index: 1902

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Press space to fire your weapon. Hold shift to reload."
    )

    # Must be a short,user-facing description of the game:
    game_description = (
        "Survive a zombie horde by strategically shooting and reloading in a top-down arena."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    ARENA_WIDTH = 500
    ARENA_HEIGHT = 300
    
    # Colors
    COLOR_BG = (20, 20, 25)
    COLOR_ARENA = (40, 40, 50)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_ZOMBIE = (255, 50, 50)
    COLOR_BULLET_TRAIL = (255, 255, 100)
    COLOR_MUZZLE_FLASH = (255, 255, 200)
    COLOR_BLOOD = (200, 0, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SCORE = (255, 223, 0)
    COLOR_HEALTH_BAR = (50, 205, 50)
    COLOR_HEALTH_BAR_BG = (139, 0, 0)

    # Game parameters
    PLAYER_SIZE = 12
    PLAYER_SPEED = 3.0
    PLAYER_MAX_HEALTH = 100
    PLAYER_MAX_AMMO = 6
    
    ZOMBIE_SIZE = 10
    ZOMBIE_SPEED = 0.75
    ZOMBIE_HEALTH = 20 # 2 hits to kill
    ZOMBIE_DAMAGE = 1.5
    ZOMBIE_COUNT = 15
    
    BULLET_DAMAGE = 10
    RELOAD_TIME_STEPS = 20 # in steps
    MAX_STEPS = 1500

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.Font(None, 28)
        self.font_score = pygame.font.Font(None, 48)
        self.font_reload = pygame.font.Font(None, 36)

        self.arena_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.ARENA_WIDTH) // 2,
            (self.SCREEN_HEIGHT - self.ARENA_HEIGHT) // 2,
            self.ARENA_WIDTH,
            self.ARENA_HEIGHT
        )
        
        self.reset()

        # Run validation check after initialization
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.arena_rect.center)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_ammo = self.PLAYER_MAX_AMMO
        self.reloading_timer = 0
        
        self.zombies = self._spawn_zombies(self.ZOMBIE_COUNT)
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed
        
        # --- Handle Player Actions ---
        movement, space_pressed, shift_held = action[0], action[1] == 1, action[2] == 1
        
        is_reloading = self.reloading_timer > 0

        # Reloading (Shift)
        if shift_held and not is_reloading and self.player_ammo < self.PLAYER_MAX_AMMO:
            self.reloading_timer = self.RELOAD_TIME_STEPS
            # sfx: reload_start.wav

        # Movement (Arrows)
        if not is_reloading:
            move_vec = pygame.Vector2(0, 0)
            if movement == 1: move_vec.y -= 1 # Up
            elif movement == 2: move_vec.y += 1 # Down
            elif movement == 3: move_vec.x -= 1 # Left
            elif movement == 4: move_vec.x += 1 # Right
            
            if move_vec.length() > 0:
                move_vec.scale_to_length(self.PLAYER_SPEED)
                self.player_pos += move_vec

        # Shooting (Space)
        if space_pressed and not is_reloading and self.player_ammo > 0:
            self.player_ammo -= 1
            # sfx: shoot.wav
            target_zombie = self._find_nearest_zombie()
            if target_zombie:
                hit, reward_bonus, score_bonus = self._process_shot(target_zombie)
                reward += reward_bonus
                self.score += score_bonus
            
        # --- Update Game State ---
        self._update_reload_timer()
        self._update_zombies()
        self._update_particles()
        
        # Player-Zombie Collision & Damage
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE/2, self.player_pos.y - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for zombie in self.zombies:
            zombie_rect = pygame.Rect(zombie['pos'].x - self.ZOMBIE_SIZE/2, zombie['pos'].y - self.ZOMBIE_SIZE/2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            if player_rect.colliderect(zombie_rect):
                self.player_health -= self.ZOMBIE_DAMAGE
                # sfx: player_hurt.wav

        # Clamp player position within arena
        self.player_pos.x = np.clip(self.player_pos.x, self.arena_rect.left + self.PLAYER_SIZE/2, self.arena_rect.right - self.PLAYER_SIZE/2)
        self.player_pos.y = np.clip(self.player_pos.y, self.arena_rect.top + self.PLAYER_SIZE/2, self.arena_rect.bottom - self.PLAYER_SIZE/2)
        
        self.steps += 1
        
        # --- Check Termination Conditions ---
        terminated = False
        if self.player_health <= 0:
            terminated = True
            reward -= 50 # Large penalty for dying
            self.game_over = True
            # sfx: game_over.wav
        elif not self.zombies:
            terminated = True
            reward += 100 # Large reward for winning
            self.game_over = True
            # sfx: victory.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _spawn_zombies(self, count):
        zombies = []
        for _ in range(count):
            while True:
                pos = pygame.Vector2(
                    self.np_random.uniform(self.arena_rect.left, self.arena_rect.right),
                    self.np_random.uniform(self.arena_rect.top, self.arena_rect.bottom)
                )
                if pos.distance_to(self.player_pos) > 100: # Don't spawn on player
                    break
            zombies.append({'pos': pos, 'health': self.ZOMBIE_HEALTH})
        return zombies

    def _find_nearest_zombie(self):
        if not self.zombies:
            return None
        return min(self.zombies, key=lambda z: self.player_pos.distance_to(z['pos']))

    def _process_shot(self, target_zombie):
        # Muzzle flash
        self.particles.append({'type': 'flash', 'pos': self.player_pos.copy(), 'radius': 15, 'lifetime': 2})
        
        # Bullet trail
        direction = (target_zombie['pos'] - self.player_pos).normalize()
        end_pos = self.player_pos + direction * 1000 # A point far away
        self.particles.append({'type': 'trail', 'start': self.player_pos.copy(), 'end': target_zombie['pos'].copy(), 'lifetime': 3})

        # Damage
        target_zombie['health'] -= self.BULLET_DAMAGE
        # sfx: zombie_hit.wav
        
        # Blood splatter
        for _ in range(15):
            offset = pygame.Vector2(self.np_random.uniform(-10, 10), self.np_random.uniform(-10, 10))
            self.particles.append({
                'type': 'blood', 'pos': target_zombie['pos'] + offset, 
                'radius': self.np_random.uniform(1, 3), 'lifetime': self.np_random.integers(10, 20)
            })

        reward = 1.0
        score = 10
        
        if target_zombie['health'] <= 0:
            self.zombies.remove(target_zombie)
            reward += 5.0
            score += 50
            # sfx: zombie_die.wav

        return True, reward, score

    def _update_reload_timer(self):
        if self.reloading_timer > 0:
            self.reloading_timer -= 1
            if self.reloading_timer == 0:
                self.player_ammo = self.PLAYER_MAX_AMMO
                # sfx: reload_complete.wav

    def _update_zombies(self):
        for zombie in self.zombies:
            direction = self.player_pos - zombie['pos']
            if direction.length() > 0:
                direction.scale_to_length(self.ZOMBIE_SPEED)
                zombie['pos'] += direction

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['lifetime'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw arena
        pygame.draw.rect(self.screen, self.COLOR_ARENA, self.arena_rect)

        # Draw particles
        for p in self.particles:
            if p['type'] == 'flash':
                alpha = int(255 * (p['lifetime'] / 2))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), self.COLOR_MUZZLE_FLASH + (alpha,))
            elif p['type'] == 'trail':
                alpha = int(255 * (p['lifetime'] / 3))
                pygame.draw.line(self.screen, self.COLOR_BULLET_TRAIL + (alpha,), p['start'], p['end'], 2)
            elif p['type'] == 'blood':
                alpha = int(255 * (p['lifetime'] / 20))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), self.COLOR_BLOOD + (alpha,))

        # Draw zombies
        for zombie in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, (zombie['pos'].x - self.ZOMBIE_SIZE/2, zombie['pos'].y - self.ZOMBIE_SIZE/2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE))

        # Draw player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (self.player_pos.x - self.PLAYER_SIZE/2, self.player_pos.y - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE))
        # Player glow effect
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), int(self.PLAYER_SIZE * 0.8), self.COLOR_PLAYER + (50,))

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_width * health_ratio, bar_height))
        
        # Ammo Count
        ammo_text = self.font_ui.render(f"AMMO: {self.player_ammo}/{self.PLAYER_MAX_AMMO}", True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (self.SCREEN_WIDTH - ammo_text.get_width() - 10, 10))

        # Score
        score_text = self.font_score.render(f"{self.score:06d}", True, self.COLOR_SCORE)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 30))
        self.screen.blit(score_text, score_rect)

        # Reloading Text
        if self.reloading_timer > 0:
            reload_text = self.font_reload.render("RELOADING...", True, self.COLOR_TEXT)
            reload_rect = reload_text.get_rect(center=(self.player_pos.x, self.player_pos.y - 30))
            self.screen.blit(reload_text, reload_rect)
        
        # Game Over Text
        if self.game_over:
            if not self.zombies:
                end_text_str = "SURVIVED"
                end_color = self.COLOR_PLAYER
            else:
                end_text_str = "GAME OVER"
                end_color = self.COLOR_ZOMBIE
            
            end_text = self.font_score.render(end_text_str, True, end_color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "zombies_remaining": len(self.zombies),
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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to run headless
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Example ---
    # This part will only work if not running headless
    try:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Zombie Survival")
        clock = pygame.time.Clock()

        obs, info = env.reset()
        done = False
        
        print(GameEnv.game_description)
        print(GameEnv.user_guide)

        while not done:
            # --- Human Input to Action ---
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_pressed = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_pressed, shift_held]
            
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- Rendering ---
            # The observation is the frame, so we just need to display it
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()

            clock.tick(30) # Match the intended FPS

        print(f"Game Over! Final Info: {info}")
        env.close()

    except pygame.error as e:
        if "No available video device" in str(e):
            print("\nPygame display unavailable (running headless). Manual play is disabled.")
            print("To run the simulation and validation, this is expected behavior.")
        else:
            raise e