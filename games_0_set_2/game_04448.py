import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to run, ↑ to jump. Reach the green exit before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape a bomb-laden level by running and jumping to avoid procedurally generated explosions within a strict time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- CONSTANTS ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.LEVEL_WIDTH = 1280
        self.FPS = 30
        self.MAX_STAGES = 3
        
        # Physics
        self.GRAVITY = 0.8
        self.PLAYER_SPEED = 6.0
        self.JUMP_STRENGTH = 15.0
        self.PLAYER_FRICTION = 0.85
        self.FLOOR_Y = self.HEIGHT - 50

        # Player
        self.PLAYER_WIDTH, self.PLAYER_HEIGHT = 20, 40

        # Bombs & Explosions
        self.BOMB_FUSE_TIME = 3 * self.FPS
        self.BOMB_RADIUS = 10
        self.EXPLOSION_RADIUS = 60
        self.EXPLOSION_DURATION = 0.5 * self.FPS

        # Timing
        self.STAGE_TIME_LIMIT = 60 * self.FPS
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_BG_LAYER_1 = (30, 35, 55)
        self.COLOR_BG_LAYER_2 = (40, 45, 65)
        self.COLOR_FLOOR = (60, 65, 85)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_EXIT = (0, 255, 128)
        self.COLOR_BOMB = np.array([255, 50, 50])
        self.COLOR_BOMB_FLASH = np.array([255, 200, 200])
        self.COLOR_EXPLOSION_1 = (255, 200, 0)
        self.COLOR_EXPLOSION_2 = (255, 100, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 20)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        if "SDL_VIDEODRIVER" not in os.environ:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.bombs = None
        self.explosions = None
        self.particles = None
        self.stage = None
        self.stage_timer = None
        self.camera_x = None
        self.camera_shake = None
        self.exit_pos = None
        self.base_bomb_spawn_prob = None
        self.last_player_x = None
        self.dodged_bombs = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # self.reset() # This is called by the validation function
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self._start_stage()
        
        return self._get_observation(), self._get_info()

    def _start_stage(self):
        self.player_pos = np.array([50.0, self.FLOOR_Y - self.PLAYER_HEIGHT])
        self.player_vel = np.array([0.0, 0.0])
        self.last_player_x = self.player_pos[0]
        self.on_ground = False
        
        self.stage_timer = self.STAGE_TIME_LIMIT
        self.bombs = []
        self.explosions = []
        self.particles = []
        self.camera_x = 0
        self.camera_shake = 0
        self.dodged_bombs = 0

        self.exit_pos = np.array([self.LEVEL_WIDTH - 60, self.FLOOR_Y - 80])
        
        # Difficulty scaling
        self.base_bomb_spawn_prob = 0.01 + (self.stage - 1) * 0.02
        
        # Initial bombs
        for _ in range(5):
            self._spawn_bomb()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- ACTION HANDLING ---
        movement, _, _ = action
        
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel[0] = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel[0] = self.PLAYER_SPEED
        else: # No-op or Down
            self.player_vel[0] *= self.PLAYER_FRICTION

        # Vertical movement (Jump)
        if movement == 1 and self.on_ground: # Up
            self.player_vel[1] = -self.JUMP_STRENGTH
            self.on_ground = False
            # Effect: Jump dust particles
            for _ in range(10):
                self.particles.append(self._create_particle(
                    pos=[self.player_pos[0] + self.PLAYER_WIDTH / 2, self.FLOOR_Y],
                    vel=[(self.np_random.random() - 0.5) * 2, -self.np_random.random() * 2],
                    life=15, color=self.COLOR_FLOOR, radius=3))

        # --- PHYSICS & STATE UPDATE ---
        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        
        # Update position
        self.player_pos += self.player_vel
        
        # Ground collision
        if self.player_pos[1] + self.PLAYER_HEIGHT >= self.FLOOR_Y:
            self.player_pos[1] = self.FLOOR_Y - self.PLAYER_HEIGHT
            self.player_vel[1] = 0
            self.on_ground = True

        # Level boundaries
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.LEVEL_WIDTH - self.PLAYER_WIDTH)

        # Camera follow
        target_camera_x = self.player_pos[0] - self.WIDTH / 2
        self.camera_x += (target_camera_x - self.camera_x) * 0.1
        self.camera_x = np.clip(self.camera_x, 0, self.LEVEL_WIDTH - self.WIDTH)
        self.camera_shake *= 0.9

        # --- GAME LOGIC ---
        self.stage_timer -= 1
        
        # Update bombs
        new_bombs = []
        for bomb in self.bombs:
            bomb['timer'] -= 1
            if bomb['timer'] <= 0:
                self._create_explosion(bomb['pos'])
                self.dodged_bombs += 1
            else:
                new_bombs.append(bomb)
        self.bombs = new_bombs

        # Update explosions
        self.explosions = [exp for exp in self.explosions if exp['timer'] > 0]
        for exp in self.explosions:
            exp['timer'] -= 1

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['vel'][1] += 0.1 # particle gravity

        # Spawn new bombs
        if self.np_random.random() < self.base_bomb_spawn_prob:
            self._spawn_bomb()

        # --- REWARD CALCULATION ---
        reward = 0.1  # Survival reward
        
        # Penalty for not moving towards exit
        if self.player_pos[0] < self.last_player_x - 0.1:
            reward -= 0.02
        self.last_player_x = self.player_pos[0]
        
        # Reward for dodging bombs
        if self.dodged_bombs > 0:
            reward += self.dodged_bombs * 1.0
            self.dodged_bombs = 0

        # --- TERMINATION CHECK ---
        terminated = False
        player_rect = pygame.Rect(self.player_pos, (self.PLAYER_WIDTH, self.PLAYER_HEIGHT))

        # 1. Collision with explosion
        for exp in self.explosions:
            dist = np.linalg.norm(player_rect.center - exp['pos'])
            if dist < self.EXPLOSION_RADIUS + player_rect.width / 2:
                terminated = True
                reward = -100
                break
        
        # 2. Timer runs out
        if self.stage_timer <= 0:
            terminated = True
            reward = -100
        
        # 3. Reached exit
        exit_rect = pygame.Rect(self.exit_pos, (40, 80))
        if not terminated and player_rect.colliderect(exit_rect):
            reward += 5 # Stage clear bonus
            if self.stage == self.MAX_STAGES:
                terminated = True
                reward += 100 # Final victory bonus
            else:
                self.stage += 1
                self._start_stage()
        
        if terminated:
            self.game_over = True

        self.score += reward
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_bomb(self):
        bomb_x = self.np_random.integers(self.camera_x, self.camera_x + self.WIDTH)
        bomb_y = self.FLOOR_Y - self.BOMB_RADIUS
        
        # Avoid spawning on player
        player_rect = pygame.Rect(self.player_pos, (self.PLAYER_WIDTH, self.PLAYER_HEIGHT))
        if player_rect.collidepoint(bomb_x, bomb_y):
            return

        self.bombs.append({
            'pos': np.array([float(bomb_x), float(bomb_y)]),
            'timer': self.BOMB_FUSE_TIME,
        })

    def _create_explosion(self, pos):
        self.explosions.append({'pos': pos.copy(), 'timer': self.EXPLOSION_DURATION})
        self.camera_shake = 15
        for _ in range(50):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 5 + 2
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            color = self.COLOR_EXPLOSION_1 if self.np_random.random() > 0.3 else self.COLOR_EXPLOSION_2
            self.particles.append(self._create_particle(pos, vel, life, color, self.np_random.integers(3, 7)))

    def _create_particle(self, pos, vel, life, color, radius):
        return {'pos': pos.copy(), 'vel': list(vel), 'life': life, 'color': color, 'radius': radius}

    def _world_to_screen(self, world_pos):
        shake_offset_x = (self.np_random.random() - 0.5) * self.camera_shake
        shake_offset_y = (self.np_random.random() - 0.5) * self.camera_shake
        return (
            int(world_pos[0] - self.camera_x + shake_offset_x),
            int(world_pos[1] + shake_offset_y)
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Parallax Background
        for i in range(5):
            # Layer 2
            pygame.draw.rect(self.screen, self.COLOR_BG_LAYER_2, 
                (i * 300 - self.camera_x * 0.5 % 300, self.FLOOR_Y - 200, 150, 200))
            # Layer 1
            pygame.draw.rect(self.screen, self.COLOR_BG_LAYER_1,
                (i * 200 - self.camera_x * 0.25 % 200, self.FLOOR_Y - 150, 100, 150))

        # Floor
        floor_screen_pos = self._world_to_screen([0, self.FLOOR_Y])
        pygame.draw.rect(self.screen, self.COLOR_FLOOR, (0, floor_screen_pos[1], self.WIDTH, self.HEIGHT - floor_screen_pos[1]))

        # Exit
        exit_screen_pos = self._world_to_screen(self.exit_pos)
        exit_rect = pygame.Rect(exit_screen_pos, (40, 80))
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, exit_rect, 3)

        # Bombs
        for bomb in self.bombs:
            pos = self._world_to_screen(bomb['pos'])
            flash_alpha = (math.cos(bomb['timer'] * 0.5) + 1) / 2
            interpolated_color = self.COLOR_BOMB * (1 - flash_alpha) + self.COLOR_BOMB_FLASH * flash_alpha
            current_color = tuple(interpolated_color.astype(int))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BOMB_RADIUS, current_color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BOMB_RADIUS, self.COLOR_PLAYER)
            # Fuse indicator
            fuse_progress = 1 - (bomb['timer'] / self.BOMB_FUSE_TIME)
            pygame.draw.arc(self.screen, self.COLOR_PLAYER, (pos[0]-self.BOMB_RADIUS, pos[1]-self.BOMB_RADIUS, self.BOMB_RADIUS*2, self.BOMB_RADIUS*2), 0, fuse_progress * 2 * math.pi, 2)

        # Particles
        for p in self.particles:
            pos = self._world_to_screen(p['pos'])
            alpha = max(0, 255 * (p['life'] / 30.0))
            color_with_alpha = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (pos[0] - p['radius'], pos[1] - p['radius']))

        # Explosions
        for exp in self.explosions:
            pos = self._world_to_screen(exp['pos'])
            progress = (self.EXPLOSION_DURATION - exp['timer']) / self.EXPLOSION_DURATION
            current_radius = int(self.EXPLOSION_RADIUS * math.sin(progress * math.pi))
            alpha = int(255 * (1 - progress))
            if current_radius > 0:
                color = self.COLOR_EXPLOSION_1 + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], current_radius, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], current_radius, color)

        # Player
        squash = max(0, -self.player_vel[1] * 0.5)
        stretch = max(0, self.player_vel[1] * 0.5)
        player_w = self.PLAYER_WIDTH + squash - stretch
        player_h = self.PLAYER_HEIGHT - squash + stretch
        player_screen_pos = self._world_to_screen(self.player_pos)
        player_rect = pygame.Rect(
            player_screen_pos[0], 
            player_screen_pos[1] + (self.PLAYER_HEIGHT - player_h),
            player_w, player_h
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
    
    def _render_ui(self):
        # Stage
        self._draw_text(f"STAGE {self.stage}", (20, 20), self.font_large, self.COLOR_TEXT)
        
        # Timer
        time_str = f"TIME: {self.stage_timer / self.FPS:.1f}"
        self._draw_text(time_str, (self.WIDTH - 150, 28), self.font_large, self.COLOR_TEXT)
        
        # Score
        score_str = f"SCORE: {int(self.score)}"
        self._draw_text(score_str, (20, 55), self.font_small, self.COLOR_TEXT)

        if self.game_over:
            outcome_text = "VICTORY!" if self.score > 0 else "GAME OVER"
            text_surf = self.font_large.render(outcome_text, True, self.COLOR_EXIT if self.score > 0 else tuple(self.COLOR_BOMB))
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _draw_text(self, text, pos, font, color):
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surf, (pos[0]+2, pos[1]+2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "time_left": self.stage_timer / self.FPS,
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        
        # Test reset
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

# This block allows running the environment directly for testing.
if __name__ == "__main__":
    
    # Test headless rendering
    print("Testing headless rendering...")
    env = GameEnv()
    obs, info = env.reset()
    assert obs.shape == (400, 640, 3)
    print("Reset successful.")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (400, 640, 3)
    print("Step successful.")
    print("Headless test passed.")

    # Interactive play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bomb Escape")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("Interactive Mode")
    print(env.user_guide)
    print("="*30)

    while running:
        # Action defaults
        movement = 0 # none
        space = 0 # released
        shift = 0 # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
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
        total_reward += reward

        # Draw the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    pygame.quit()