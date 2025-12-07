
# Generated: 2025-08-27T13:29:08.399899
# Source Brief: brief_00382.md
# Brief Index: 382

        
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
    """
    A top-down target practice game where players must strategically expend
    limited ammunition to destroy all targets within a time limit.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the reticle. Press Space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Arcade target shooter. Clear all 20 targets with 25 shots before the timer runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 900  # 30 seconds at 30 FPS
        self.NUM_TARGETS = 20
        self.INITIAL_AMMO = 25
        self.RETICLE_SPEED = 15
        self.PROJECTILE_SPEED = 25
        self.TARGET_MIN_RADIUS = 10
        self.TARGET_MAX_RADIUS = 20
        self.TARGET_MIN_SPAWN_DIST = 50 # Min distance between target centers

        # --- Colors ---
        self.COLOR_BG = (10, 15, 25)
        self.COLOR_TARGET = (255, 60, 60)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_UI = (200, 200, 255)
        self.COLOR_RETICLE = (0, 255, 255)
        self.COLOR_EXPLOSION = (255, 150, 50)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 60, bold=True)

        # --- State Variables ---
        self.reticle_pos = None
        self.targets = []
        self.projectiles = []
        self.particles = []
        self.ammo = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.np_random = None

        self.reset()
        # self.validate_implementation() # Uncomment to run validation check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.reticle_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.ammo = self.INITIAL_AMMO
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = True  # Prevent firing on first frame of a new episode
        self.projectiles.clear()
        self.particles.clear()
        self._generate_targets()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        # Unpack and process action
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement)
        
        # Check for fire event (press, not hold)
        fired_this_step = space_held and not self.prev_space_held
        if fired_this_step and self.ammo > 0:
            self._fire_projectile()
            reward -= 0.1  # Small cost for firing to discourage spam
        self.prev_space_held = space_held

        # Update game state
        self._update_projectiles()
        self._update_particles()

        # Check for collisions and calculate hit rewards
        reward += self._check_collisions()

        # Check for termination and add terminal rewards
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _generate_targets(self):
        self.targets.clear()
        for _ in range(self.NUM_TARGETS):
            attempts = 0
            while attempts < 100:
                radius = self.np_random.integers(self.TARGET_MIN_RADIUS, self.TARGET_MAX_RADIUS + 1)
                pos = pygame.Vector2(
                    self.np_random.integers(radius, self.WIDTH - radius),
                    self.np_random.integers(radius, self.HEIGHT - radius - 60) # Keep top clear for UI
                )
                if all(pos.distance_to(t['pos']) > (t['radius'] + radius + 10) for t in self.targets):
                    self.targets.append({'pos': pos, 'radius': int(radius)})
                    break
                attempts += 1

    def _handle_input(self, movement):
        if movement == 1: self.reticle_pos.y -= self.RETICLE_SPEED
        elif movement == 2: self.reticle_pos.y += self.RETICLE_SPEED
        elif movement == 3: self.reticle_pos.x -= self.RETICLE_SPEED
        elif movement == 4: self.reticle_pos.x += self.RETICLE_SPEED
        self.reticle_pos.x = np.clip(self.reticle_pos.x, 0, self.WIDTH)
        self.reticle_pos.y = np.clip(self.reticle_pos.y, 0, self.HEIGHT)

    def _fire_projectile(self):
        self.ammo -= 1
        start_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT)
        try:
            direction = (self.reticle_pos - start_pos).normalize()
        except ValueError: # If reticle is exactly at start_pos
            direction = pygame.Vector2(0, -1)
        # sound: laser_shoot.wav
        self.projectiles.append({'pos': start_pos, 'vel': direction * self.PROJECTILE_SPEED})

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p['pos'] += p['vel']
            if not (0 <= p['pos'].x <= self.WIDTH and 0 <= p['pos'].y <= self.HEIGHT):
                self.projectiles.remove(p)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_collisions(self):
        hit_reward = 0
        for p in self.projectiles[:]:
            for t in self.targets[:]:
                if p['pos'].distance_to(t['pos']) < t['radius']:
                    self.score += 10
                    hit_reward += 5.0
                    self._create_explosion(t['pos'], t['radius'])
                    self.targets.remove(t)
                    # sound: explosion.wav
                    if p in self.projectiles:
                        self.projectiles.remove(p)
                    break 
        return hit_reward

    def _create_explosion(self, pos, radius):
        num_particles = int(radius * 1.5)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 25),
                'max_life': 25,
                'size': self.np_random.integers(2, 5)
            })

    def _check_termination(self):
        if not self.targets:
            return True, 100.0  # Win condition
        if self.ammo <= 0 and not self.projectiles:
            return True, -50.0  # Lose: out of ammo
        if self.steps >= self.MAX_STEPS:
            return True, -50.0  # Lose: out of time
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for t in self.targets:
            pos_int = (int(t['pos'].x), int(t['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, *pos_int, t['radius'], self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, *pos_int, t['radius'], self.COLOR_TARGET)

        for p in self.projectiles:
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, pos_int, 3)

        for part in self.particles:
            pos_int = (int(part['pos'].x), int(part['pos'].y))
            life_ratio = part['life'] / part['max_life']
            alpha = int(255 * life_ratio)
            color = (*self.COLOR_EXPLOSION, alpha)
            temp_surf = pygame.Surface((part['size']*2, part['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (part['size'], part['size']), part['size'])
            self.screen.blit(temp_surf, (pos_int[0] - part['size'], pos_int[1] - part['size']))
        
        self._render_reticle()

    def _render_reticle(self):
        x, y = int(self.reticle_pos.x), int(self.reticle_pos.y)
        size = 15
        pygame.gfxdraw.aacircle(self.screen, x, y, size + 5, (*self.COLOR_RETICLE, 50))
        pygame.gfxdraw.aacircle(self.screen, x, y, size + 3, (*self.COLOR_RETICLE, 70))
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (x - size, y), (x + size, y), 2)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (x, y - size), (x, y + size), 2)
        pygame.gfxdraw.aacircle(self.screen, x, y, size // 2, self.COLOR_RETICLE)

    def _render_ui(self):
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.metadata["render_fps"])
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI)
        self.screen.blit(time_text, (10, 10))

        ammo_text = self.font_ui.render(f"AMMO: {self.ammo}", True, self.COLOR_UI)
        self.screen.blit(ammo_text, (self.WIDTH - ammo_text.get_width() - 10, 10))

        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))
        
        targets_text = self.font_ui.render(f"TARGETS: {len(self.targets)}", True, self.COLOR_UI)
        self.screen.blit(targets_text, (10, self.HEIGHT - targets_text.get_height() - 10))
        
        if self.game_over:
            win = not self.targets
            msg = "MISSION COMPLETE" if win else "MISSION FAILED"
            color = (50, 255, 50) if win else (255, 50, 50)
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ammo": self.ammo,
            "targets_left": len(self.targets)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    # You will need to install pygame for this to work: pip install pygame
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(env.game_description)
    
    terminated = False
    clock = pygame.time.Clock()
    
    # Game loop
    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

        # --- Environment Step ---
        obs, reward, terminated, _, info = env.step(action)
        
        # --- Rendering ---
        # Pygame uses a different coordinate system, so we need to transpose
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.metadata["render_fps"])

    env.close()
    pygame.quit()