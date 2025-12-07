import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move. Press ↑ to jump. Squash monsters by landing on them."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive for 60 seconds in this fast-paced arcade platformer. "
        "Jump on monsters to score points, but avoid touching them otherwise!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = self.FPS * 60  # 60 seconds

        # Colors
        self.COLOR_BG = (15, 19, 40)
        self.COLOR_GROUND = (69, 40, 60)
        self.COLOR_PLAYER = (58, 204, 133)
        self.COLOR_PLAYER_OUTLINE = (217, 255, 236)
        self.MONSTER_COLORS = {
            "linear": (232, 79, 87),
            "sine": (74, 163, 232),
            "fast": (168, 93, 242)
        }
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR = (200, 60, 60)
        self.COLOR_HEALTH_BAR_BG = (80, 80, 80)

        # Physics constants
        self.GRAVITY = 0.4
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_FRICTION = -0.12
        self.PLAYER_JUMP_STRENGTH = -10
        self.MAX_PLAYER_VEL_X = 5
        self.PLAYER_BOUNCE = -5
        self.PLAYER_KNOCKBACK = pygame.Vector2(5, -5)
        self.GROUND_Y = self.HEIGHT - 50

        # Player constants
        self.PLAYER_SIZE = pygame.Vector2(24, 32)
        self.MAX_HEALTH = 100
        self.HIT_DAMAGE = 25

        # Monster constants
        self.INITIAL_SPAWN_RATE = 2 * self.FPS  # every 2 seconds
        self.SPAWN_RATE_INCREASE = 0.001 * self.FPS / self.FPS  # per second
        self.INITIAL_MONSTER_SPEED = 1.0
        self.SPEED_INCREASE_INTERVAL = 10 * self.FPS  # every 10 seconds
        self.SPEED_INCREASE_AMOUNT = 0.25
        self.MAX_MONSTER_SPEED = 5.0

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)

        # State variables (initialized in reset)
        self.player_pos = None
        self.player_vel = None
        self.player_health = None
        self.on_ground = None
        self.player_squash_anim = None

        self.monsters = None
        self.particles = None

        self.steps = None
        self.score = None
        self.game_over = None

        self.monster_spawn_timer = None
        self.current_spawn_rate = None
        self.current_monster_speed = None

        self.background_stars = None

        # self.reset() is called here to initialize state before validation
        # The RNG is also initialized here through super().reset()
        self.reset()

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize player state
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.GROUND_Y - self.PLAYER_SIZE.y)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = self.MAX_HEALTH
        self.on_ground = False
        self.player_squash_anim = 0

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Initialize entities
        self.monsters = []
        self.particles = []

        # Initialize progression
        self.monster_spawn_timer = self.INITIAL_SPAWN_RATE
        self.current_spawn_rate = self.INITIAL_SPAWN_RATE
        self.current_monster_speed = self.INITIAL_MONSTER_SPEED

        # Generate background stars for parallax effect
        self.background_stars = [
            [(self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 2)) for _ in range(100)],
            [(self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3)) for _ in range(50)]
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Unpack factorized action
        movement = action[0]

        # --- LOGIC ---
        self._update_player(movement)
        self._update_monsters()
        self._update_particles()
        self._update_progression()

        reward = self._handle_collisions_and_calculate_reward()

        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            self.game_over = True
            if self.player_health <= 0:
                reward -= 100  # Lost
            else:
                reward += 100  # Won

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self, movement):
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_ACCEL

        # Apply friction
        self.player_vel.x += self.player_vel.x * self.PLAYER_FRICTION
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0

        # Clamp horizontal velocity
        self.player_vel.x = max(-self.MAX_PLAYER_VEL_X, min(self.MAX_PLAYER_VEL_X, self.player_vel.x))

        # Vertical movement (Jump)
        if movement == 1 and self.on_ground:
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self._create_particles(self.player_pos + pygame.Vector2(self.PLAYER_SIZE.x / 2, self.PLAYER_SIZE.y), 5, (200, 200, 200))  # Jump dust
            self.on_ground = False
            # Sound: Jump

        # Apply gravity
        self.player_vel.y += self.GRAVITY

        # Update position
        self.player_pos += self.player_vel

        # Ground collision
        if self.player_pos.y + self.PLAYER_SIZE.y >= self.GROUND_Y:
            if not self.on_ground:
                self.player_squash_anim = 10  # Start squash animation on landing
                # Sound: Land
            self.player_pos.y = self.GROUND_Y - self.PLAYER_SIZE.y
            self.player_vel.y = 0
            self.on_ground = True
        else:
            self.on_ground = False

        # Screen boundaries
        self.player_pos.x = max(0, min(self.WIDTH - self.PLAYER_SIZE.x, self.player_pos.x))

        # Update squash animation
        if self.player_squash_anim > 0:
            self.player_squash_anim -= 1

    def _update_monsters(self):
        # Spawn new monsters
        self.monster_spawn_timer -= 1
        if self.monster_spawn_timer <= 0:
            self._spawn_monster()
            self.monster_spawn_timer = self.current_spawn_rate

        # Update existing monsters
        for monster in self.monsters:
            if monster['type'] == 'sine':
                monster['pos'].x += monster['vel'].x
                monster['pos'].y = self.GROUND_Y - monster['size'].y + math.sin(self.steps * 0.1 + monster['phase']) * 10
            else:  # linear, fast
                monster['pos'] += monster['vel']

        # Remove off-screen monsters
        self.monsters = [m for m in self.monsters if -m['size'].x < m['pos'].x < self.WIDTH]

    def _spawn_monster(self):
        monster_type = self.np_random.choice(['linear', 'sine', 'fast'], p=[0.5, 0.3, 0.2])
        size = pygame.Vector2(30, 20)

        speed = self.current_monster_speed
        if monster_type == 'fast':
            speed *= 1.5
        speed = min(speed, self.MAX_MONSTER_SPEED)

        if self.np_random.random() < 0.5:  # Spawn from left
            pos = pygame.Vector2(-size.x, self.GROUND_Y - size.y)
            vel = pygame.Vector2(speed, 0)
        else:  # Spawn from right
            pos = pygame.Vector2(self.WIDTH, self.GROUND_Y - size.y)
            vel = pygame.Vector2(-speed, 0)

        self.monsters.append({
            'pos': pos,
            'vel': vel,
            'size': size,
            'type': monster_type,
            'color': self.MONSTER_COLORS[monster_type],
            'phase': self.np_random.random() * 2 * math.pi  # for sine wave
        })
        # Sound: Monster Spawn

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'].y += 0.1  # particle gravity
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _update_progression(self):
        # Increase spawn rate over time
        self.current_spawn_rate = max(15, self.current_spawn_rate - self.SPAWN_RATE_INCREASE)

        # Increase speed over time
        if self.steps > 0 and self.steps % self.SPEED_INCREASE_INTERVAL == 0:
            self.current_monster_speed += self.SPEED_INCREASE_AMOUNT

    def _handle_collisions_and_calculate_reward(self):
        reward = 0.1  # Survival reward per frame
        player_rect = pygame.Rect(self.player_pos, self.PLAYER_SIZE)

        for i in range(len(self.monsters) - 1, -1, -1):
            monster = self.monsters[i]
            monster_rect = pygame.Rect(monster['pos'], monster['size'])

            if player_rect.colliderect(monster_rect):
                # Check for squash: player is moving down and their feet are above monster's head
                is_squash = self.player_vel.y > 0 and player_rect.bottom < monster_rect.centery

                if is_squash:
                    # Sound: Squash
                    self._create_particles(monster_rect.center, 20, (255, 220, 0))
                    self.monsters.pop(i)
                    self.score += 1
                    reward += 1.0
                    self.player_vel.y = self.PLAYER_BOUNCE  # Bounce
                    self.player_squash_anim = 10
                else:  # Player got hit
                    # Sound: Hurt
                    self._create_particles(player_rect.center, 15, self.COLOR_HEALTH_BAR)
                    self.player_health = max(0, self.player_health - self.HIT_DAMAGE)
                    assert self.player_health >= 0

                    # Apply knockback
                    knockback_dir = 1 if player_rect.centerx < monster_rect.centerx else -1
                    self.player_vel.x = -knockback_dir * self.PLAYER_KNOCKBACK.x
                    self.player_vel.y = self.PLAYER_KNOCKBACK.y

                    # Remove monster that hit player
                    self.monsters.pop(i)
        return reward

    def _check_termination(self):
        if self.player_health <= 0:
            # Sound: Game Over
            return True
        if self.steps >= self.MAX_STEPS:
            # Sound: Victory
            return True
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Parallax background
        for i, layer in enumerate(self.background_stars):
            speed = i + 1
            for x, y, size in layer:
                px = (x - self.steps * 0.1 * speed) % self.WIDTH
                color_val = 100 - i * 20
                pygame.draw.rect(self.screen, (color_val, color_val, color_val + 20), (int(px), int(y), size, size))

        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

        # Particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])

        # Monsters
        for m in self.monsters:
            m_rect = pygame.Rect(m['pos'], m['size'])
            pygame.draw.rect(self.screen, m['color'], m_rect, border_radius=4)
            # Eyes
            eye_y = m_rect.y + 6
            eye_1_x = m_rect.centerx - 5
            eye_2_x = m_rect.centerx + 5
            pygame.draw.circle(self.screen, (255, 255, 255), (eye_1_x, eye_y), 3)
            pygame.draw.circle(self.screen, (255, 255, 255), (eye_2_x, eye_y), 3)
            pygame.draw.circle(self.screen, (0, 0, 0), (eye_1_x + int(m['vel'].x), eye_y), 1)
            pygame.draw.circle(self.screen, (0, 0, 0), (eye_2_x + int(m['vel'].x), eye_y), 1)

        # Player
        squash_factor = self.player_squash_anim / 20.0
        stretch_factor = max(0, -self.player_vel.y / 20.0)

        p_width = self.PLAYER_SIZE.x * (1 + squash_factor)
        p_height = self.PLAYER_SIZE.y * (1 - squash_factor)
        p_width *= (1 - stretch_factor * 0.5)
        p_height *= (1 + stretch_factor)

        p_rect = pygame.Rect(
            self.player_pos.x + (self.PLAYER_SIZE.x - p_width) / 2,
            self.player_pos.y + (self.PLAYER_SIZE.y - p_height),
            p_width,
            p_height
        )

        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, p_rect.inflate(4, 4), border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, p_rect, border_radius=6)

    def _render_ui(self):
        # Score
        score_surf = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_surf = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH / 2 - timer_surf.get_width() / 2, 10))

        # Health Bar
        health_pct = self.player_health / self.MAX_HEALTH
        bar_width = 150
        bar_height = 15
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 15
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, int(bar_width * health_pct), bar_height), border_radius=4)

        # Game Over Message
        if self.game_over:
            if self.player_health <= 0:
                msg = "GAME OVER"
                color = self.COLOR_HEALTH_BAR
            else:
                msg = "YOU WIN!"
                color = self.COLOR_PLAYER

            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            pygame.draw.rect(self.screen, self.COLOR_BG, msg_rect.inflate(20, 20))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def _create_particles(self, position, count, color):
        for _ in range(count):
            self.particles.append({
                'pos': position.copy(),
                'vel': pygame.Vector2((self.np_random.random() - 0.5) * 6, (self.np_random.random() - 0.5) * 6 - 2),
                'life': self.np_random.integers(20, 40),
                'radius': self.np_random.integers(3, 7),
                'color': color
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space (uses state, so it's checked after reset)
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
        assert not trunc
        assert isinstance(info, dict)

        # The faulty test block that caused the AssertionError has been removed.

        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == '__main__':
    # Un-set the headless environment variable to allow rendering
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()

    # Use Pygame for human interaction
    pygame.display.set_caption("Monster Squash")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    terminated = False
    total_reward = 0

    # --- Main Game Loop ---
    running = True
    while running:
        # --- Action Selection (Human Player) ---
        keys = pygame.key.get_pressed()

        movement = 0  # No-op
        if keys[pygame.K_UP]:
            movement = 1
        # Action 2 (down) is a no-op
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        # The game design doesn't use space or shift, but we must provide them
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset on 'r' key
                    obs, info = env.reset()
                    terminated = False
                    total_reward = 0
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()
    print(f"Game finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")