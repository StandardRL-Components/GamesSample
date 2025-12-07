
# Generated: 2025-08-27T18:57:10.375710
# Source Brief: brief_01998.md
# Brief Index: 1998

        
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
        "Controls: Arrows determine jump direction. Space for a normal jump, Shift for a high jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Explore procedural crystal caverns. Leap across platforms and collect 50 crystals to win before falling into the abyss."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRAVITY = 0.7
        self.PLAYER_JUMP_NORMAL = 12
        self.PLAYER_JUMP_HIGH = 16
        self.PLAYER_HORZ_SPEED = 7
        self.MAX_STEPS = 1500
        self.WIN_CRYSTALS = 50

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PIT = (10, 12, 20)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_PLAYER_GLOW = (*self.COLOR_PLAYER, 50)
        self.COLOR_PLATFORM = (80, 90, 110)
        self.COLOR_PLATFORM_TOP = (150, 160, 180)
        self.COLOR_CRYSTAL = (100, 200, 255)
        self.COLOR_CRYSTAL_GLOW = (*self.COLOR_CRYSTAL, 80)
        self.COLOR_TEXT = (230, 230, 240)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 32)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # State variables (initialized in reset)
        self.player_pos = None
        self.player_vel = None
        self.player_size = None
        self.on_ground = None
        self.player_jump_anim_timer = None
        self.platforms = None
        self.crystals = None
        self.particles = None
        self.bg_elements = None
        self.camera_offset_x = None
        self.steps = None
        self.score = None
        self.crystals_collected = None
        self.game_over = None
        self.game_won = None
        self.last_player_x = None
        
        # Initialize state variables
        self.reset()

        # Run self-check
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = pygame.Vector2(100, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_size = pygame.Vector2(20, 30)
        self.on_ground = False
        self.player_jump_anim_timer = 0
        self.last_player_x = self.player_pos.x

        # Game state
        self.steps = 0
        self.score = 0
        self.crystals_collected = 0
        self.game_over = False
        self.game_won = False
        
        # World state
        self.camera_offset_x = 0
        self.platforms = []
        self.crystals = []
        self.particles = []
        self._generate_bg_elements()
        self._generate_initial_world()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            # If game is over, observe but don't act. Return terminal state.
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle Input
        self._handle_input(action)

        # 2. Update Physics & State
        self._update_physics()
        self._handle_collisions()
        
        # 3. Manage World
        self._manage_world_generation()
        self._update_camera()
        self._update_particles()
        
        # 4. Calculate Reward
        reward = self._calculate_reward()

        # 5. Check Termination
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.game_won:
                reward += 100.0
            else:
                reward -= 100.0

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if self.on_ground:
            jump_power = 0
            if shift_held:
                jump_power = self.PLAYER_JUMP_HIGH
            elif space_held:
                jump_power = self.PLAYER_JUMP_NORMAL

            if jump_power > 0:
                self.player_vel.y = -jump_power
                self.on_ground = False
                self.player_jump_anim_timer = 10 # For squash/stretch effect
                # sound: jump.wav
                self._create_particles(self.player_pos + pygame.Vector2(self.player_size.x / 2, self.player_size.y), 10, self.COLOR_PLATFORM_TOP)

                horz_vel_map = {
                    1: 0, # Up
                    2: 0, # Down (results in a short vertical hop)
                    3: -self.PLAYER_HORZ_SPEED, # Left
                    4: self.PLAYER_HORZ_SPEED, # Right
                }
                self.player_vel.x = horz_vel_map.get(movement, 0)
                if movement == 2: # Down makes a weaker jump
                    self.player_vel.y *= 0.6

    def _update_physics(self):
        # Apply gravity
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY
            self.player_vel.y = min(self.player_vel.y, 15) # Terminal velocity

        # Air friction
        self.player_vel.x *= 0.98

        # Update position
        self.player_pos += self.player_vel

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos, self.player_size)
        
        # Check for landing on platforms
        newly_grounded = False
        for plat in self.platforms:
            if player_rect.colliderect(plat) and self.player_vel.y >= 0:
                # Check if player was above the platform in the previous frame
                if (player_rect.bottom - self.player_vel.y) <= plat.top + 1:
                    self.player_pos.y = plat.top - self.player_size.y
                    self.player_vel.y = 0
                    self.player_vel.x *= 0.8 # Ground friction
                    newly_grounded = True
                    break # Stop after finding one platform to stand on
        
        if newly_grounded and not self.on_ground:
            # sound: land.wav
            self._create_particles(self.player_pos + pygame.Vector2(self.player_size.x / 2, self.player_size.y), 5, self.COLOR_PLATFORM_TOP)
        self.on_ground = newly_grounded

        # Crystal collection
        collected_indices = []
        for i, crystal in enumerate(self.crystals):
            if player_rect.colliderect(crystal['rect']):
                collected_indices.append(i)
                self.crystals_collected += 1
                self.score += 10
                # sound: collect_crystal.wav
                self._create_particles(crystal['rect'].center, 20, self.COLOR_CRYSTAL)
        
        for i in sorted(collected_indices, reverse=True):
            self.crystals.pop(i)

    def _calculate_reward(self):
        reward = 0.0
        # Reward for collecting crystals is event-based and added in _handle_collisions
        # We can add it here instead to centralize reward logic.
        # Let's stick to the brief: score and reward are linked.
        # The score is updated in collision, so the reward is just the delta.
        # However, the brief asks for +10 per crystal, let's do that explicitly.
        # The score update is fine for info, but reward should be explicit.
        # A simple way is to count collected crystals in _handle_collisions and add reward here.
        # Let's adjust: _handle_collisions will just remove crystals and increment self.crystals_collected.
        # The reward logic will be here. Let's assume `_handle_collisions` was just run.
        # Okay, let's just use the score for simplicity, since it's a 1-to-1 mapping.
        
        # Reward for moving right
        dx = self.player_pos.x - self.last_player_x
        if dx > 0:
            reward += dx * 0.1
        self.last_player_x = self.player_pos.x

        # Check score for crystal reward
        # This is a bit tricky, let's keep it simple: reward is added in _handle_collisions
        # by simply adding to the step reward. Let's refactor that.

        step_reward = reward # Store movement reward
        # Re-do collision to add reward cleanly
        collected_indices = []
        player_rect = pygame.Rect(self.player_pos, self.player_size)
        for i, crystal in enumerate(self.crystals):
            if crystal['rect'].colliderect(player_rect):
                collected_indices.append(i)

        if collected_indices:
            step_reward += 10.0 * len(collected_indices)
            self.crystals_collected += len(collected_indices)
            self.score += 10 * len(collected_indices)
            for i in sorted(collected_indices, reverse=True):
                crystal = self.crystals.pop(i)
                # sound: collect_crystal.wav
                self._create_particles(crystal['rect'].center, 20, self.COLOR_CRYSTAL)

        return step_reward

    def _check_termination(self):
        if self.crystals_collected >= self.WIN_CRYSTALS:
            self.game_won = True
            return True
        if self.player_pos.y > self.HEIGHT + 50: # Fell in pit
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _update_camera(self):
        target_offset_x = self.player_pos.x - self.WIDTH / 3
        self.camera_offset_x += (target_offset_x - self.camera_offset_x) * 0.1

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_world()
        self._render_particles()
        self._render_player()
        
        # Render UI overlay
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crystals": self.crystals_collected,
        }

    # --- RENDER METHODS ---
    def _render_background(self):
        self.screen.fill(self.COLOR_PIT, (0, self.HEIGHT - 20, self.WIDTH, 20))
        self.screen.fill(self.COLOR_BG)

        for element in self.bg_elements:
            x = (element['pos'][0] - self.camera_offset_x * element['depth']) % self.WIDTH
            y = element['pos'][1]
            pygame.draw.circle(self.screen, element['color'], (int(x), int(y)), element['size'])

    def _render_world(self):
        # Render platforms
        for plat in self.platforms:
            screen_rect = plat.move(-self.camera_offset_x, 0)
            top_rect = pygame.Rect(screen_rect.left, screen_rect.top, screen_rect.width, 4)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_TOP, top_rect, border_radius=3)

        # Render crystals
        for crystal in self.crystals:
            screen_rect = crystal['rect'].move(-self.camera_offset_x, 0)
            anim_prog = (math.sin(self.steps * 0.1 + crystal['anim_offset']) + 1) / 2
            
            # Pulsing size
            size_mod = 1.0 + 0.2 * anim_prog
            center = screen_rect.center
            w, h = screen_rect.width * size_mod, screen_rect.height * size_mod
            pulse_rect = pygame.Rect(center[0] - w/2, center[1] - h/2, w, h)
            
            # Draw glow
            glow_surf = pygame.Surface((w*2, h*2), pygame.SRCALPHA)
            pygame.draw.ellipse(glow_surf, self.COLOR_CRYSTAL_GLOW, glow_surf.get_rect())
            self.screen.blit(glow_surf, (pulse_rect.centerx - w, pulse_rect.centery - h), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Draw crystal
            pygame.draw.ellipse(self.screen, self.COLOR_CRYSTAL, pulse_rect)

    def _render_player(self):
        current_size = self.player_size.copy()
        if self.player_jump_anim_timer > 0:
            p = self.player_jump_anim_timer / 10.0
            current_size.x *= (1.0 - 0.3 * p)
            current_size.y *= (1.0 + 0.4 * p)
            self.player_jump_anim_timer -= 1
        
        screen_pos_x = int(self.player_pos.x - self.camera_offset_x)
        screen_pos_y = int(self.player_pos.y)
        
        player_rect = pygame.Rect(
            screen_pos_x - (current_size.x - self.player_size.x) / 2,
            screen_pos_y - (current_size.y - self.player_size.y),
            int(current_size.x),
            int(current_size.y)
        )

        # Glow effect
        glow_surf = pygame.Surface((player_rect.width * 2, player_rect.height * 2), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=int(6 * (player_rect.width / self.player_size.x)))
        self.screen.blit(glow_surf, (player_rect.centerx - player_rect.width, player_rect.centery - player_rect.height), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_ui(self):
        score_text = f"SCORE: {self.score}"
        crystals_text = f"CRYSTALS: {self.crystals_collected} / {self.WIN_CRYSTALS}"
        
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        crystals_surf = self.font_ui.render(crystals_text, True, self.COLOR_TEXT)
        
        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(crystals_surf, (10, 40))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        text = "YOU WIN!" if self.game_won else "GAME OVER"
        text_surf = self.font_game_over.render(text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    # --- PARTICLE SYSTEM ---
    def _create_particles(self, pos, count, color):
        for _ in range(count):
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(random.uniform(-3, 3), random.uniform(-3, 3)),
                'life': random.randint(10, 20),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.9
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            screen_pos = p['pos'] - pygame.Vector2(self.camera_offset_x, 0)
            size = max(0, p['life'] * 0.2)
            pygame.draw.circle(self.screen, p['color'], (int(screen_pos.x), int(screen_pos.y)), int(size))

    # --- WORLD GENERATION ---
    def _generate_bg_elements(self):
        self.bg_elements = []
        for _ in range(100):
            self.bg_elements.append({
                'pos': (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT)),
                'size': random.randint(1, 2),
                'depth': random.uniform(0.1, 0.5),
                'color': random.choice([(50, 60, 80), (40, 50, 70)])
            })

    def _generate_initial_world(self):
        start_platform = pygame.Rect(self.player_pos.x - 40, self.player_pos.y + 60, 100, 20)
        self.platforms.append(start_platform)
        self._manage_world_generation()
    
    def _manage_world_generation(self):
        # Generate new platforms if needed
        if not self.platforms or self.platforms[-1].right < self.camera_offset_x + self.WIDTH + 200:
            last_x = self.platforms[-1].right if self.platforms else 0
            last_y = self.platforms[-1].centery if self.platforms else self.HEIGHT / 2 + 80
            self._generate_platforms_and_crystals(last_x, last_y)

        # Clean up off-screen elements
        self.platforms = [p for p in self.platforms if p.right > self.camera_offset_x - 50]
        self.crystals = [c for c in self.crystals if c['rect'].right > self.camera_offset_x - 50]

    def _generate_platforms_and_crystals(self, start_x, start_y):
        current_x = start_x
        current_y = start_y

        for _ in range(10): # Generate a chunk of 10 platforms
            difficulty_mod = self.crystals_collected // 10
            min_gap = 40 + difficulty_mod * 5
            max_gap = min(150, 100 + difficulty_mod * 10)
            
            gap = self.np_random.uniform(min_gap, max_gap)
            width = self.np_random.uniform(80, 200)
            
            current_x += gap
            delta_y = self.np_random.uniform(-80, 80)
            current_y = np.clip(current_y + delta_y, self.HEIGHT * 0.4, self.HEIGHT - 50)
            
            new_platform = pygame.Rect(int(current_x), int(current_y), int(width), 20)
            self.platforms.append(new_platform)

            # Add a crystal?
            if self.np_random.random() < 0.6:
                crystal_pos_x = new_platform.centerx
                crystal_pos_y = new_platform.top - self.np_random.uniform(30, 80)
                crystal_rect = pygame.Rect(int(crystal_pos_x - 10), int(crystal_pos_y - 10), 20, 20)
                self.crystals.append({'rect': crystal_rect, 'anim_offset': self.np_random.random() * math.pi * 2})

            current_x += width

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to run and display the game
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Override render_mode for human playback
    env.metadata["render_modes"] = ["human"]
    pygame.display.set_caption("Crystal Caverns")
    human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    done = False
    while not done:
        # Human controls
        keys = pygame.key.get_pressed()
        mov = 0 # none
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Display the frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

    pygame.quit()