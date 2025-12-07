import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move. Survive the monster onslaught for 60 seconds."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a monster onslaught for 60 seconds by dodging enemies and collecting power-ups in a top-down arcade environment."
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
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_timer = pygame.font.Font(None, 48)

        # Game constants
        self.MAX_STEPS = 1800  # 60 seconds at 30fps
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 5
        self.INITIAL_MONSTER_COUNT = 20
        self.POWERUP_COUNT = 5
        self.MONSTER_SIZE = 18
        self.POWERUP_SIZE = 10

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_INVINCIBLE = (255, 255, 128)
        self.MONSTER_COLORS = [(255, 50, 50), (50, 100, 255), (255, 255, 50)]
        self.COLOR_POWERUP = (0, 255, 255)
        self.COLOR_UI = (255, 255, 255)

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.invincibility_timer = None
        self.monsters = None
        self.powerups = None
        self.steps = None
        self.score = None
        self.game_over = None

        # self.reset() is called here, but we need the RNG first.
        # super().reset() will initialize the np_random generator.
        # We don't need a full reset() call here, as the user of the env will call it.
        # However, to pass the original code's validation logic, we'll keep it.
        # The key is to initialize self.steps before it's used.
        # A full reset is needed to initialize all attributes.
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        # These must be initialized before helper functions like _spawn_monster are called
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = 3
        self.invincibility_timer = 0
        
        self.monsters = []
        for _ in range(self.INITIAL_MONSTER_COUNT):
            self._spawn_monster()
            
        self.powerups = []
        for _ in range(self.POWERUP_COUNT):
            self._spawn_powerup()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0.01  # Small reward for surviving a frame

        # Update game logic
        self._handle_player_movement(movement)
        self._update_monsters()
        
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1

        # Handle collisions and rewards
        reward += self._handle_collisions()
        
        # Proximity penalty
        is_invincible = self.invincibility_timer > 0
        if not is_invincible:
            for m in self.monsters:
                if self.player_pos.distance_to(m['pos']) < self.PLAYER_SIZE * 3:
                    reward -= 0.02 # Encourage dodging
                    break

        self.steps += 1
        
        # Check for termination conditions
        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.player_health > 0: # Survived the time limit
                reward += 50.0 # Victory bonus
                # sfx: victory_fanfare.wav

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_monster(self):
        # Spawn away from the center
        edge = self.np_random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -self.MONSTER_SIZE)
        elif edge == 'bottom':
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.MONSTER_SIZE)
        elif edge == 'left':
            pos = pygame.Vector2(-self.MONSTER_SIZE, self.np_random.uniform(0, self.HEIGHT))
        else: # right
            pos = pygame.Vector2(self.WIDTH + self.MONSTER_SIZE, self.np_random.uniform(0, self.HEIGHT))

        speed_multiplier = 1.0 + (self.steps / 300) * 0.1
        base_speed = self.np_random.uniform(1.0, 2.0)

        self.monsters.append({
            'pos': pos,
            'color': self.MONSTER_COLORS[self.np_random.integers(0, len(self.MONSTER_COLORS))],
            'speed': base_speed * speed_multiplier,
            'patrol_axis': self.np_random.choice(['h', 'v']),
            'patrol_dir': self.np_random.choice([-1, 1]),
            'drift_speed': self.np_random.uniform(0.1, 0.3)
        })

    def _spawn_powerup(self, position=None):
        if position is None:
            pos = pygame.Vector2(
                self.np_random.uniform(self.POWERUP_SIZE, self.WIDTH - self.POWERUP_SIZE),
                self.np_random.uniform(self.POWERUP_SIZE, self.HEIGHT - self.POWERUP_SIZE)
            )
        else:
            pos = pygame.Vector2(position)

        self.powerups.append({'pos': pos})

    def _handle_player_movement(self, movement):
        if movement == 1:  # Up
            self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos.y += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos.x += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE / 2, self.WIDTH - self.PLAYER_SIZE / 2)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE / 2, self.HEIGHT - self.PLAYER_SIZE / 2)
    
    def _update_monsters(self):
        speed_multiplier = 1.0 + (self.steps / 300) * 0.1
        for m in self.monsters:
            # Main patrol movement
            if m['patrol_axis'] == 'h':
                m['pos'].x += m['speed'] * m['patrol_dir'] * speed_multiplier
                if m['pos'].x < -self.MONSTER_SIZE or m['pos'].x > self.WIDTH + self.MONSTER_SIZE:
                    m['patrol_dir'] *= -1
            else: # 'v'
                m['pos'].y += m['speed'] * m['patrol_dir'] * speed_multiplier
                if m['pos'].y < -self.MONSTER_SIZE or m['pos'].y > self.HEIGHT + self.MONSTER_SIZE:
                    m['patrol_dir'] *= -1
            
            # Drift towards player
            if (self.player_pos - m['pos']).length() > 0:
                direction_to_player = (self.player_pos - m['pos']).normalize()
                if m['patrol_axis'] == 'h':
                    m['pos'].y += direction_to_player.y * m['drift_speed'] * speed_multiplier
                else:
                    m['pos'].x += direction_to_player.x * m['drift_speed'] * speed_multiplier

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Player-monster collision
        if self.invincibility_timer <= 0:
            for m in self.monsters:
                monster_rect = pygame.Rect(m['pos'].x - self.MONSTER_SIZE / 2, m['pos'].y - self.MONSTER_SIZE / 2, self.MONSTER_SIZE, self.MONSTER_SIZE)
                if player_rect.colliderect(monster_rect):
                    self.player_health -= 1
                    # sfx: player_hit.wav
                    self.invincibility_timer = 60 # 2 seconds of post-hit invincibility
                    reward -= 5.0
                    assert self.player_health >= 0
                    break # Only one hit per frame
        
        # Player-powerup collision
        for i, p in enumerate(self.powerups):
            powerup_rect = pygame.Rect(p['pos'].x - self.POWERUP_SIZE, p['pos'].y - self.POWERUP_SIZE, self.POWERUP_SIZE * 2, self.POWERUP_SIZE * 2)
            if player_rect.colliderect(powerup_rect):
                self.score += 10
                # sfx: powerup_collect.wav
                self.invincibility_timer = max(self.invincibility_timer, 30) # 1 second invincibility
                reward += 1.0
                self.powerups.pop(i)
                self._spawn_powerup()
                assert len(self.powerups) <= self.POWERUP_COUNT
                break

        return reward

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render monsters
        for m in self.monsters:
            pygame.draw.rect(self.screen, m['color'], (int(m['pos'].x - self.MONSTER_SIZE / 2), int(m['pos'].y - self.MONSTER_SIZE / 2), self.MONSTER_SIZE, self.MONSTER_SIZE))

        # Render powerups with glow
        for p in self.powerups:
            pos_x, pos_y = int(p['pos'].x), int(p['pos'].y)
            glow_radius = int(self.POWERUP_SIZE * 1.8)
            # Use gfxdraw for smooth circles
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, glow_radius, (*self.COLOR_POWERUP, 50))
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, glow_radius, (*self.COLOR_POWERUP, 80))
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, self.POWERUP_SIZE, self.COLOR_POWERUP)
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, self.POWERUP_SIZE, self.COLOR_POWERUP)

        # Render player
        is_invincible = self.invincibility_timer > 0
        player_visible = True
        if is_invincible:
            # Flash effect when invincible
            if (self.steps // 3) % 2 == 0:
                player_visible = False
            color = self.COLOR_PLAYER_INVINCIBLE
            # Shield effect
            shield_radius = int(self.PLAYER_SIZE * 0.8)
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), shield_radius, (*color, 60))
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), shield_radius, (*color, 100))
        else:
            color = self.COLOR_PLAYER
        
        if player_visible:
            pygame.draw.rect(self.screen, color, (int(self.player_pos.x - self.PLAYER_SIZE / 2), int(self.player_pos.y - self.PLAYER_SIZE / 2), self.PLAYER_SIZE, self.PLAYER_SIZE))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Health
        health_str = "♥" * self.player_health + "♡" * (3 - self.player_health)
        health_text = self.font_ui.render(f"Health: {health_str}", True, self.COLOR_UI)
        health_rect = health_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(health_text, health_rect)

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / 30)
        assert time_left <= 60.01 # Add tolerance for float precision
        timer_text = self.font_timer.render(f"{time_left:.1f}", True, self.COLOR_UI)
        timer_rect = timer_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 30))
        self.screen.blit(timer_text, timer_rect)

    def _get_info(self):
        assert self.score >= 0
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "time_left": max(0, (self.MAX_STEPS - self.steps) / 30)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Validating implementation...")
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
    # For this to work, you might need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a display window
    pygame.display.set_caption(env.game_description)
    screen_display = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    total_reward = 0
    
    while not done:
        # --- Human Controls ---
        movement_action = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement_action = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement_action = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement_action = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement_action, space_action, shift_action]
        # ----------------------

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # Limit frame rate
        if env.auto_advance:
            clock.tick(30)
    
    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()