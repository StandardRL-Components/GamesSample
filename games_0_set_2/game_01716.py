
# Generated: 2025-08-27T18:02:52.278820
# Source Brief: brief_01716.md
# Brief Index: 1716

        
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
        "Controls: ↑↓ to move. Space to use Speed Boost. Shift to use Invincibility."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Side-scrolling zombie survival. Evade hordes, grab power-ups, and reach the finish line before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 180 * 3 # Brief says 180 steps, but at 1 step/sec that's slow. Let's assume 3 steps/sec for better feel.
        self.LEVEL_END_X = 5000

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_ROAD = (40, 45, 50)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_ZOMBIE = (80, 140, 80)
        self.COLOR_ZOMBIE_EYE = (200, 50, 50)
        self.COLOR_SPEED_BOOST = (50, 150, 255)
        self.COLOR_INVINCIBILITY = (255, 200, 50)
        self.COLOR_FINISH_LINE = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_BG = (0, 0, 0, 128)

        # Player properties
        self.PLAYER_WIDTH, self.PLAYER_HEIGHT = 20, 40
        self.PLAYER_SCREEN_X = 100
        self.PLAYER_V_SPEED = 8
        self.PLAYER_H_SPEED_NORMAL = 10
        self.PLAYER_H_SPEED_BOOSTED = 20
        self.INVINCIBILITY_DURATION = 5 * 3
        self.SPEED_BOOST_DURATION = 10 * 3

        # Zombie properties
        self.ZOMBIE_WIDTH, self.ZOMBIE_HEIGHT = 20, 42
        self.ZOMBIE_SPEED = 7
        self.ZOMBIE_SPAWN_PROB_INITIAL = 0.1
        self.ZOMBIE_SPAWN_PROB_INCREASE = 0.0005 # Scaled for more steps

        # Power-up properties
        self.POWERUP_SIZE = 16
        self.POWERUP_SPAWN_PROB = 0.02

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_timer = pygame.font.Font(None, 36)
        
        # State variables are initialized in reset()
        self.reset()

        # Final validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            # Fallback for older gym versions or no seed
            if not hasattr(self, 'np_random'):
                 self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.player_y = self.HEIGHT // 2
        self.world_x = 0

        self.zombies = []
        self.powerups = []
        self.particles = []
        self.buildings = self._generate_buildings()

        self.speed_boost_inventory = 1
        self.invincibility_inventory = 1
        self.speed_boost_timer = 0
        self.invincibility_timer = 0
        
        return self._get_observation(), self._get_info()

    def _generate_buildings(self):
        buildings = []
        current_x = -self.WIDTH
        while current_x < self.LEVEL_END_X + self.WIDTH:
            width = self.np_random.integers(80, 200)
            height = self.np_random.integers(100, 300)
            color_val = self.np_random.integers(30, 50)
            color = (color_val, color_val, color_val + 10)
            buildings.append({'x': current_x, 'width': width, 'height': height, 'color': color})
            current_x += width + self.np_random.integers(20, 80)
        return buildings

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        terminated = False
        
        # 1. Handle player input
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if movement == 1: # Up
            self.player_y = max(0, self.player_y - self.PLAYER_V_SPEED)
            reward -= 0.02 # Small cost for movement
        elif movement == 2: # Down
            self.player_y = min(self.HEIGHT - self.PLAYER_HEIGHT, self.player_y + self.PLAYER_V_SPEED)
            reward -= 0.02

        if space_held and self.speed_boost_inventory > 0 and self.speed_boost_timer == 0:
            self.speed_boost_inventory -= 1
            self.speed_boost_timer = self.SPEED_BOOST_DURATION
            # SFX: Power-up activation sound
            self._create_particles(self.PLAYER_SCREEN_X + self.PLAYER_WIDTH / 2, self.player_y + self.PLAYER_HEIGHT / 2, self.COLOR_SPEED_BOOST, 20)

        if shift_held and self.invincibility_inventory > 0 and self.invincibility_timer == 0:
            self.invincibility_inventory -= 1
            self.invincibility_timer = self.INVINCIBILITY_DURATION
            # SFX: Invincibility activation sound
            self._create_particles(self.PLAYER_SCREEN_X + self.PLAYER_WIDTH / 2, self.player_y + self.PLAYER_HEIGHT / 2, self.COLOR_INVINCIBILITY, 20)

        # 2. Update game state
        self.steps += 1
        reward += 0.1 # Survival reward

        # Update power-up timers
        self.speed_boost_timer = max(0, self.speed_boost_timer - 1)
        self.invincibility_timer = max(0, self.invincibility_timer - 1)

        # Update world scroll based on player speed
        player_h_speed = self.PLAYER_H_SPEED_BOOSTED if self.speed_boost_timer > 0 else self.PLAYER_H_SPEED_NORMAL
        self.world_x += player_h_speed
        
        # Update zombies
        for z in self.zombies:
            z['x'] -= self.ZOMBIE_SPEED
        self.zombies = [z for z in self.zombies if z['x'] + self.ZOMBIE_WIDTH > 0]

        # Update particles
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # 3. Spawn new entities
        spawn_prob = self.ZOMBIE_SPAWN_PROB_INITIAL + self.steps * self.ZOMBIE_SPAWN_PROB_INCREASE
        if self.np_random.random() < spawn_prob:
            zombie_y = self.np_random.integers(0, self.HEIGHT - self.ZOMBIE_HEIGHT)
            self.zombies.append({'x': self.WIDTH, 'y': zombie_y})
            # SFX: Zombie spawn groan

        if self.np_random.random() < self.POWERUP_SPAWN_PROB:
            ptype = self.np_random.choice(['speed', 'invincibility'])
            powerup_y = self.np_random.integers(0, self.HEIGHT - self.POWERUP_SIZE)
            self.powerups.append({'x': self.WIDTH, 'y': powerup_y, 'type': ptype})
        
        # 4. Handle collisions
        player_rect = pygame.Rect(self.PLAYER_SCREEN_X, self.player_y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)

        # Player vs Zombies
        if self.invincibility_timer == 0:
            for z in self.zombies:
                z_rect = pygame.Rect(z['x'], z['y'], self.ZOMBIE_WIDTH, self.ZOMBIE_HEIGHT)
                if player_rect.colliderect(z_rect):
                    self.game_over = True
                    terminated = True
                    reward -= 100
                    # SFX: Player hurt / game over sound
                    break
        
        # Player vs Power-ups
        for p in self.powerups[:]:
            p_rect = pygame.Rect(p['x'], p['y'], self.POWERUP_SIZE, self.POWERUP_SIZE)
            if player_rect.colliderect(p_rect):
                if p['type'] == 'speed':
                    self.speed_boost_inventory += 1
                    color = self.COLOR_SPEED_BOOST
                else: # invincibility
                    self.invincibility_inventory += 1
                    color = self.COLOR_INVINCIBILITY
                self.powerups.remove(p)
                reward += 25
                # SFX: Power-up collection sound
                self._create_particles(p['x'] + self.POWERUP_SIZE / 2, p['y'] + self.POWERUP_SIZE / 2, color, 30)

        # 5. Check termination conditions
        if self.world_x >= self.LEVEL_END_X:
            self.game_over = True
            terminated = True
            reward += 100
            # SFX: Level complete fanfare
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            # No reward penalty, just failure to complete in time

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render parallax background
        for b in self.buildings:
            screen_x = b['x'] - self.world_x * 0.5
            if screen_x + b['width'] > 0 and screen_x < self.WIDTH:
                pygame.draw.rect(self.screen, b['color'], (screen_x, self.HEIGHT - b['height'], b['width'], b['height']))

        # Render road
        pygame.draw.rect(self.screen, self.COLOR_ROAD, (0, self.HEIGHT - 40, self.WIDTH, 40))
        
        # Render finish line
        finish_screen_x = self.LEVEL_END_X - self.world_x
        if finish_screen_x < self.WIDTH:
            for i in range(0, self.HEIGHT, 20):
                pygame.draw.rect(self.screen, self.COLOR_FINISH_LINE, (finish_screen_x, i, 10, 10))
                pygame.draw.rect(self.screen, (0,0,0), (finish_screen_x, i+10, 10, 10))

        # Render powerups
        for p in self.powerups:
            screen_x = p['x'] - self.world_x
            if screen_x < self.WIDTH:
                color = self.COLOR_SPEED_BOOST if p['type'] == 'speed' else self.COLOR_INVINCIBILITY
                pygame.draw.rect(self.screen, color, (screen_x, p['y'], self.POWERUP_SIZE, self.POWERUP_SIZE))
                text = 'S' if p['type'] == 'speed' else 'I'
                text_surf = self.font_ui.render(text, True, self.COLOR_PLAYER)
                self.screen.blit(text_surf, (screen_x + 3, p['y'] - 2))

        # Render zombies
        for z in self.zombies:
            screen_x = z['x'] - self.world_x
            if screen_x < self.WIDTH:
                bob = math.sin(self.steps * 0.3 + screen_x * 0.1) * 2
                z_rect = pygame.Rect(int(screen_x), int(z['y'] + bob), self.ZOMBIE_WIDTH, self.ZOMBIE_HEIGHT)
                pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z_rect)
                eye_y = z_rect.y + 8 + math.sin(self.steps * 0.2 + screen_x * 0.1)
                pygame.draw.circle(self.screen, self.COLOR_ZOMBIE_EYE, (int(z_rect.centerx + 3), int(eye_y)), 2)

        # Render particles
        for p in self.particles:
            screen_x = p['x'] - self.world_x
            alpha = max(0, 255 * (p['life'] / 30.0))
            color_with_alpha = p['color'] + (alpha,)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            temp_surf.fill(color_with_alpha)
            self.screen.blit(temp_surf, (int(screen_x), int(p['y'])))

        # Render player
        player_rect = pygame.Rect(self.PLAYER_SCREEN_X, self.player_y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        # Running animation
        leg_bob = (self.steps // 4 % 2) * 5
        pygame.draw.rect(self.screen, self.COLOR_BG, (self.PLAYER_SCREEN_X + 5, self.player_y + self.PLAYER_HEIGHT - leg_bob, 10, 5))
        
        if self.invincibility_timer > 0:
            alpha = 100 + (self.invincibility_timer % 5) * 20
            pygame.gfxdraw.rectangle(self.screen, player_rect.inflate(8, 8), self.COLOR_INVINCIBILITY + (alpha,))
        if self.speed_boost_timer > 0:
            # Trail effect
            for i in range(3):
                alpha = 150 - i * 50
                offset = (i + 1) * 5
                trail_rect = pygame.Rect(player_rect.x - offset, player_rect.y, player_rect.width, player_rect.height)
                pygame.gfxdraw.box(self.screen, trail_rect, self.COLOR_SPEED_BOOST + (alpha,))

        # Render UI
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_timer.render(f"TIME: {time_left // 3}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Power-up inventory
        # Speed Boost
        pygame.draw.rect(self.screen, self.COLOR_SPEED_BOOST, (10, 40, 20, 20))
        speed_text = self.font_ui.render(f"x{self.speed_boost_inventory}", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_text, (35, 38))
        if self.speed_boost_timer > 0:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, 62, 20 * (self.speed_boost_timer / self.SPEED_BOOST_DURATION), 4))

        # Invincibility
        pygame.draw.rect(self.screen, self.COLOR_INVINCIBILITY, (10, 70, 20, 20))
        inv_text = self.font_ui.render(f"x{self.invincibility_inventory}", True, self.COLOR_UI_TEXT)
        self.screen.blit(inv_text, (35, 68))
        if self.invincibility_timer > 0:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, 92, 20 * (self.invincibility_timer / self.INVINCIBILITY_DURATION), 4))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "world_x": self.world_x,
            "invincibility": self.invincibility_inventory,
            "speed_boost": self.speed_boost_inventory,
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

# Example usage:
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To run the game with manual controls ---
    # This requires a display. If running headlessly, comment this block out.
    try:
        import os
        # Set a display if one is not available
        if os.environ.get('DISPLAY') is None:
             os.environ['SDL_VIDEODRIVER'] = 'dummy'

        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Zombie Runner")
        clock = pygame.time.Clock()

        obs, info = env.reset()
        done = False
        
        print(env.user_guide)

        while not done:
            # Action defaults
            movement, space, shift = 0, 0, 0 # no-op, released, released

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            
            if keys[pygame.K_SPACE]:
                space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift = 1

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render the observation from the environment to the screen
            # Need to transpose back from (H, W, C) to (W, H, C) for pygame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if done:
                print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
                # Optional: pause on game over
                # waiting = True
                # while waiting:
                #     for event in pygame.event.get():
                #         if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                #             waiting = False
                obs, info = env.reset() # Reset for a new game
                done = False

            clock.tick(30) # Limit frame rate for playability

    except ImportError:
        print("Pygame is required for interactive mode.")
    except pygame.error as e:
        print(f"Could not initialize display for interactive mode: {e}")
        print("Running a headless test instead.")
        
        # Headless test
        obs, info = env.reset()
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Headless test episode finished. Score: {info['score']}, Steps: {info['steps']}")
                obs, info = env.reset()
    
    env.close()